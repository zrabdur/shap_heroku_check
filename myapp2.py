import streamlit as st
import streamlit.components.v1 as components
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#
from numpy.random import RandomState
import re

from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, auc
from sklearn.model_selection import train_test_split

import hyperopt
from catboost import CatBoostClassifier, Pool, cv
from catboost.utils import get_confusion_matrix, get_roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import KNeighborsClassifier

#@st.cache(allow_output_mutation=True)
@st.cache
    
def getJensenShannon(df, responseFeature, targetFeature):
    """
    Input  : Dataframe, Control Feature (List of a string), Target Feature Name (List of a string)
    Output : Two-way Interaction for Reponse Leakage Testing 
    Output : Jensen-Shannon Distance Measure (Average KL Divergence)
    """
    p = df.groupby(responseFeature).size().to_list()
    p = [k/df.shape[0] for k in p]

    q = df.groupby(targetFeature).size().to_list()
    q = [k/df.shape[0] for k in q]
    
    del_len = len(q)-len(p)
    
    if del_len > 0:
        p = p + [0.0]*(del_len)
        js_distance = jensenshannon(p, q, base=2)
    elif del_len < 0:
        print("Jensen-Shannon distance measure can not be performed")
        js_distance = np.nan
    else:
        js_distance = jensenshannon(p, q, base=2)
    
    return js_distance
#--------------------------------------------------------------------------
st.title("""Airline passenger data model:		""")
# Reading data
#df = pd.read_excel(r"C:\Users\AR\data_customerSatisfaction.xlsx")
df = pd.read_excel(r"C:\Users\AR\data_customerSatisfaction_sampleData_100_rows.xlsx")
st.write("DataFrame")
#
# Processing data
disp_df = df.head(5)
st.write(disp_df)
print("Shape of the Dataframe :", df.shape)
st.write("Shape of the Dataframe :", df.shape)
print(df.columns.to_list())
# st.write("Dataframe Columns :", df.columns.to_list())
#
# Fixing column names, removing White Spaces and Slashes
old_Name = df.columns.to_list()
new_Name = [re.sub(" ", "_", name.title()) if len(name.split()) > 1 else name for name in old_Name  ]
col_Name = {old_Name[i]: new_Name[i] for i in range(len(old_Name))} 
#
df = df.rename(columns=col_Name)
df = df.rename(columns={"Departure/Arrival_Time_Convenient":"Departure_Arrival_Time_Convenient"})
# print( df.dtypes)
#
#______________________________________________________________________________________________
# Column id does not have any implication, dropping column id
df = df.drop(["id"], axis=1)  
#
# Replacing null values of Age to median value
medianAge = df.loc[df['Age'].isnull()==False, 'Age'].median()
df.loc[df['Age'].isnull()==True, 'Age'] = medianAge
# Leg_Room_Service has some outlier value 44, 45 etc. Changing values larger than 5 to 3.
df.loc[ df["Leg_Room_Service"] > 5, "Leg_Room_Service"] = 3.0

#Replacing Food and Drink null values by 1
df.loc[ df["Food_And_Drink"].isnull()==True, "Food_And_Drink" ] = 1.0

#Replacing Online Boarding null values by 1
df.loc[ df["Online_Boarding"].isnull()==True, "Online_Boarding" ] = 1.0

# Replacing outlier Flight Distance values by median value
distList = sorted( df["Flight_Distance"].unique(), reverse= True)
df.loc[ df["Flight_Distance"] > distList[1],  "Flight_Distance"] = df["Flight_Distance"].median()

#Replacing Baggage_Handling null values by 1
df.loc[ df["Baggage_Handling"].isnull() == True, "Baggage_Handling" ] = 1.0

# Departure Delay in Minutes and Arrival Delay in Minutes are Strongly Correlated (r ~ 0.99)
# Replacing Arrival Delay by Departure Delay
df = df.drop(["Arrival_Delay_In_Minutes"], axis=1)
#______________________________________________________________________________________________
#______________________________________________________________________________________________

#converting int 1,2,3-- instead of int 1.0, 2.0, 3.0 etch
cols = ['Food_And_Drink', 'Online_Boarding', 'Baggage_Handling']
df[cols] = df[cols].applymap(np.int64)
#Converting Age, Flight Distance and Departure Delay in Minutes to float64
cols_float = ['Age','Flight_Distance','Departure_Delay_In_Minutes']
df[cols_float] = df[cols_float].applymap(np.float64)
#______________________________________________________________________________________________

# Converting the Dataframe to create a dataframe using the feature names of df
df_feature = df.dtypes.to_frame().reset_index().rename(columns={"index": "Feature", 0: 'FeatureType'})
theObjects = df_feature[ df_feature['FeatureType'] == 'object']
df_object  = df[theObjects['Feature']].copy() 

# print( df.dtypes)


# Creating a sub-dataframe from the float type data columns of df
df_float = df[['Age','Flight_Distance','Departure_Delay_In_Minutes']].copy()
nFloat = df_float.shape[1]

object_float_list = df_object.columns.to_list() + df_float.columns.to_list() #Columns of float type and object type Combined
colList = set( df.columns.to_list() ) # Columns of df
ofList  = set( object_float_list ) # Columns of object and float type 
colList.difference_update( ofList ) # Difference to obtain list of int type columns
df_int = df[ colList ].copy()
df_str = df_int.astype('str') # low cardinality (<20) so converted to string
#______________________________________________________________________________________________

# Converting Categorical to Integer Type 
#0 = "Business"
#1 = "Economy and Economy plus"
df_object['Y'] = df_object.apply(lambda row: 0 if row.Class=="Business" else 1, axis=1)
# Checking class distribution
df_object.groupby('Y').size()

#Dropping Class, it is replaced by feature Y"""
df_object = df_object.drop(['Class'], axis=1)

# Final Dataframe
df = pd.concat([df_object, df_str, df_float], axis=1)
df = df.sample(frac=1.0, replace=False, random_state=1)
print(f'Shape of the final dataframe : {df.shape}')

#______________________________________________________________________________________________

# Response Leakage,if KL Divergence == 0 or JensenShannon == 0 then "Leakage" else "No Leakage"

colName = df.columns[np.where( df.dtypes != float )]

print("Printing Jensen-Shannon Distance between Y and Categorical Features ...") 

for col in [i for i in colName if i != 'Y']:
    controlFeature = ["Y"]
    targetFeature  = [col]
    js = getJensenShannon(df, controlFeature, targetFeature) 
    print(f"Y vs {col} : {round(js,4)}")

# Building Model
y = df.Y
X = df.drop('Y', axis=1)

#______________________________________________________________________________________________

#______________________________________________________________________________________________

# Categorical Features Declaration
total_features = list(range(0, X.shape[1]))
print(f'Total features : {total_features}' )
print(f'Remove {nFloat} numeric features ...' )
categorical_features_indices   = total_features[:-nFloat]
print(f'Categorical features : {categorical_features_indices}')
#______________________________________________________________________________________________

# Checking for Label Balance in the Dataset
# Class Labels & Weights
# 0 = "Business" (negative class)
# 1 = "Economy and Economy plus" (positive class)
print(f'Class Labels: {set(y)}')
print(f'Zero Count = {len(y) - sum(y)}, One Count = {sum(y)}')
zero_frac = (len(y)-sum(y))/len(y)
one_frac  =  sum(y)/len(y)
print(f'Class Distributions 0:1 : {round(zero_frac,4), round(one_frac,4)}')

# Create Class Weight
zero_weight = one_frac
one_weight  = zero_frac
print(f'Class Weights : {round(zero_weight,4), round(one_weight,4)}')
class_weight = np.array([zero_weight if x==0 else one_weight for x in y])

#______________________________________________________________________________________________
#Data Split Train (80) - Test (20)
theSeed = 201 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=theSeed)
print(f'train data: {X_train.shape}')
print(f'test data : {X_test.shape}')
#______________________________________________________________________________________________

# Creating Pool for Convenience
train_pool = Pool(X_train, y_train, cat_features=categorical_features_indices)
test_pool  = Pool(X_test, y_test, cat_features=categorical_features_indices)

#______________________________________________________________________________________________
# Train and Fit Model
model = CatBoostClassifier(
    custom_loss=['Accuracy'],
    random_seed=theSeed,
    loss_function='Logloss',
    #loss_function='MultiClass',
    logging_level='Silent')

fit = model.fit(
    X_train, y_train,
    cat_features=categorical_features_indices,
    eval_set=(X_test, y_test),
#    logging_level='Silent',  
    plot=False)


evaluate_test = fit.eval_metrics(test_pool, ['AUC'], plot=False)

theAUC = np.mean( np.array(evaluate_test['AUC']) )
print(f"The AUC : { round(theAUC,4) }")

confusion_matrix = get_confusion_matrix(fit, test_pool)
print( confusion_matrix )
st.write(confusion_matrix)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(round(accuracy,4))
st.write(round(accuracy,4))


feature_importances = fit.get_feature_importance(train_pool)
feature_names = X_train.columns
for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print('{}: {}'.format(name, round(score,4)))
    st.write('{}: {}'.format(name, round(score,4)))

#______________________________________________________________________________________________
def st_shap(plot, height=600):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)
#______________________________________________________________________________________________

explainer = shap.Explainer(fit)
shap_values = explainer(X_test)
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
#https://discuss.streamlit.io/t/display-shap-diagrams-with-streamlit/1029/9
#st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]))

#st_shap(shap.force_plot(shap_values[0,:]))
p = shap.summary_plot(shap_values, show=False) 
display(p)
 
#st_shap(shap.plots.waterfall(shap_values[0]))
#shap.plots.beeswarm(shap_values, max_display=20)
#shap_values
# fpr, tpr, thresholds = get_roc_curve(fit, test_pool, plot=True)

#_______________________________________________________________________________________________
#______________________________________________________________________________________________
#______________________________________________________________________________________________
#______________________________________________________________________________________________


