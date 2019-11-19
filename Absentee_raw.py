#Goal: Preprocess the Data to Predict Excessive Employee absence
#%%
#Import Libraries
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler

#%%
raw_csv_data= pd.read_csv('Absenteeism-data.csv')
print(raw_csv_data)
#%%
df= raw_csv_data.copy()
print(display(df))
#%%
pd.options.display.max_columns=None
pd.options.display.max_rows=None
print(display(df))
#%%
print(df.info())
#%%
df=df.drop(['ID'], axis=1)

#%%
print(display(df.head()))

#%%
#Our goal is to see who is more likely to be absent. Let's define
#our targets from our dependent variable, Absenteeism Time in Hours
print(df['Absenteeism Time in Hours'])
print(df['Absenteeism Time in Hours'].median())
#%%
targets= np.where(df['Absenteeism Time in Hours']>df['Absenteeism Time in Hours'].median(),1,0)
#%%
print(targets)
#%%
df['Excessive Absenteeism']= targets
#%%
print(df.head())

#%%
#Let's Separate the Day and Month Values to see if there is correlation
#between Day of week/month with absence
print(type(df['Date'][0]))
#%%
df['Date']= pd.to_datetime(df['Date'], format='%d/%m/%Y')
#%%
print(df['Date'])
print(type(df['Date'][0]))
#%%
#Extracting the Month Value
print(df['Date'][0].month)
#%%
list_months=[]
print(list_months)
#%%
print(df.shape)
#%%
for i in range(df.shape[0]):
    list_months.append(df['Date'][i].month)
#%%
print(list_months)
#%%
print(len(list_months))
#%%
#Let's Create a Month Value Column for df
df['Month Value']= list_months
#%%
print(df.head())
#%%
#Now let's extract the day of the week from date
df['Date'][699].weekday()
#%%
def date_to_weekday(date_value):
    return date_value.weekday()
#%%
df['Day of the Week']= df['Date'].apply(date_to_weekday)
#%%
print(df.head())
#%%
df= df.drop(['Date'], axis=1)
#%%
print(df.columns.values)
#%%
reordered_columns= ['Reason for Absence', 'Month Value','Day of the Week','Transportation Expense', 'Distance to Work', 'Age',
 'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children', 'Pets',
 'Absenteeism Time in Hours', 'Excessive Absenteeism']
#%%
df=df[reordered_columns]
print(df.head())
#%%
#First Checkpoint
df_date_mod= df.copy()
#%%
print(df_date_mod)

#%%
#Let's Standardize our inputs, ignoring the Reasons and Education Columns
#Because they are labelled by a separate categorical criteria, not numerically
print(df_date_mod.columns.values)
#%%
unscaled_inputs= df_date_mod.loc[:, ['Month Value','Day of the Week','Transportation Expense','Distance to Work','Age','Daily Work Load Average','Body Mass Index','Children','Pets','Absenteeism Time in Hours']]
#%%
print(display(unscaled_inputs))
#%%
absenteeism_scaler= StandardScaler()
#%%
absenteeism_scaler.fit(unscaled_inputs)
#%%
scaled_inputs= absenteeism_scaler.transform(unscaled_inputs)
#%%
print(display(scaled_inputs))
#%%
print(scaled_inputs.shape)
#%%
scaled_inputs= pd.DataFrame(scaled_inputs, columns=['Month Value','Day of the Week','Transportation Expense','Distance to Work','Age','Daily Work Load Average','Body Mass Index','Children','Pets','Absenteeism Time in Hours'])
print(display(scaled_inputs))
#%%
df_date_mod= df_date_mod.drop(['Month Value','Day of the Week','Transportation Expense','Distance to Work','Age','Daily Work Load Average','Body Mass Index','Children','Pets','Absenteeism Time in Hours'], axis=1)
print(display(df_date_mod))
#%%
df_date_mod=pd.concat([df_date_mod,scaled_inputs], axis=1)
print(display(df_date_mod))
#%%
df_date_mod= df_date_mod[reordered_columns]
print(display(df_date_mod.head()))
#%%
#Checkpoint
df_date_scale_mod= df_date_mod.copy()
print(display(df_date_scale_mod.head()))
#%%
#Let's Analyze the Reason for Absence Category
print(df_date_scale_mod['Reason for Absence'])
#%%
print(df_date_scale_mod['Reason for Absence'].min())
print(df_date_scale_mod['Reason for Absence'].max())
#%%
print(df_date_scale_mod['Reason for Absence'].unique())
#%%
print(len(df_date_scale_mod['Reason for Absence'].unique()))
#%%
print(sorted(df['Reason for Absence'].unique()))
#%%
reason_columns= pd.get_dummies(df['Reason for Absence'])
print(reason_columns)
#%%
reason_columns['check']= reason_columns.sum(axis=1)
print(reason_columns)
#%%
print(reason_columns['check'].sum(axis=0))
#%%
print(reason_columns['check'].unique())
#%%
reason_columns=reason_columns.drop(['check'], axis=1)
print(reason_columns)
#%%
reason_columns=pd.get_dummies(df_date_scale_mod['Reason for Absence'], drop_first=True)
print(reason_columns)
#%%
print(df_date_scale_mod.columns.values)
#%%
print(reason_columns.columns.values)
#%%
df_date_scale_mod= df_date_scale_mod.drop(['Reason for Absence'], axis=1)
print(df_date_scale_mod)
#%%
reason_type_1= reason_columns.loc[:, 1:14].max(axis=1)
reason_type_2= reason_columns.loc[:, 15:17].max(axis=1)
reason_type_3= reason_columns.loc[:, 18:21].max(axis=1)
reason_type_4= reason_columns.loc[:, 22:].max(axis=1)
#%%
print(reason_type_1)
print(reason_type_2)
print(reason_type_3)
print(reason_type_4)
#%%
print(df_date_scale_mod.head())
#%%
df_date_scale_mod= pd.concat([df_date_scale_mod, reason_type_1,reason_type_2, reason_type_3, reason_type_4], axis=1)
print(df_date_scale_mod.head())
#%%
print(df_date_scale_mod.columns.values)
#%%
column_names= ['Month Value','Day of the Week','Transportation Expense',
 'Distance to Work','Age','Daily Work Load Average','Body Mass Index',
 'Education','Children','Pets','Absenteeism Time in Hours',
 'Excessive Absenteeism', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']

df_date_scale_mod.columns= column_names
print(df_date_scale_mod.head())
#%%
column_names_reordered= ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Month Value','Day of the Week','Transportation Expense',
 'Distance to Work','Age','Daily Work Load Average','Body Mass Index',
 'Education','Children','Pets','Absenteeism Time in Hours',
 'Excessive Absenteeism']

df_date_scale_mod=df_date_scale_mod[column_names_reordered]
print(display(df_date_scale_mod.head()))
#%%
#Checkpoint
df_date_scale_mod_reas= df_date_scale_mod.copy()
print(df_date_scale_mod_reas.head())
#%%
#Let's Look at the Education column now
print(df_date_scale_mod_reas['Education'].unique())
#This shows us that education is rated from 1-4 based on level
#of completion
#%%
print(df_date_scale_mod_reas['Education'].value_counts())
#The overwhelming majority of workers are highschool educated, while the 
#rest have higher degrees
#%%
#We'll create our dummy variables as highschool and higher education
df_date_scale_mod_reas['Education']= df_date_scale_mod_reas['Education'].map({1:0, 2:1, 3:1, 4:1})
#%%
print(df_date_scale_mod_reas['Education'].unique())
#%%
print(df_date_scale_mod_reas['Education'].value_counts())
#%%
#Checkpoint
df_preprocessed= df_date_scale_mod_reas.copy()
print(display(df_preprocessed.head()))
#%%
#%%
#Split Inputs from targets
scaled_inputs_all= df_preprocessed.loc[:,'Reason_1':'Absenteeism Time in Hours']
print(display(scaled_inputs_all.head()))
print(scaled_inputs_all.shape)
#%%
targets_all= df_preprocessed.loc[:,'Excessive Absenteeism']
print(display(targets_all.head()))
print(targets_all.shape)
#%%
#Shuffle Inputs and targets
shuffled_indices= np.arange(scaled_inputs_all.shape[0])
np.random.shuffle(shuffled_indices)
shuffled_inputs= scaled_inputs_all.iloc[shuffled_indices]
shuffled_targets= targets_all[shuffled_indices]
#%%
print(shuffled_inputs.shape)
print(shuffled_targets.shape)
#%%
#Split Data into Training, Validation and Test, using the 80/10/10
samples_count= shuffled_inputs.shape[0]
train_samples_count= int(0.8*samples_count)
validation_samples_count= int(0.1*samples_count)
test_samples_count= samples_count-train_samples_count-validation_samples_count
#%%
train_inputs= shuffled_inputs[:train_samples_count]
train_targets= shuffled_targets[:train_samples_count]

validation_inputs= shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets=shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

test_inputs= shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets= shuffled_targets[train_samples_count+validation_samples_count:]
#%%
#Let's make sure our dataset is balanced
print(np.sum(train_targets), train_samples_count,np.sum(train_targets)/train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets)/validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets)/test_samples_count)
#%%
#Save Datasets as NPZ
np.savez('Absentee_data_train', inputs=train_inputs, targets=train_targets)
np.savez('Absentee_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('Absentee_data_test', inputs=test_inputs, targets=test_targets) 
