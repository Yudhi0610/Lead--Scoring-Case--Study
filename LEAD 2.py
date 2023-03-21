#!/usr/bin/env python
# coding: utf-8

# # Lead Scoring Case Study
# 
# 

# # Step 1 
# Loading And Understanding The Data

# In[2]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[3]:


# Importing data
df = pd.read_csv(r"C:\Users\user\AppData\Local\Temp\Temp1_Lead+Scoring+Case+Study.zip\Lead Scoring Assignment\Leads.csv")
df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# # Step 2
# # EXPLORATORY DATA ANALYSIS
# 
# Data Cleaning

# In[7]:


df.drop(['Lead Number', 'Prospect ID'], 1, inplace = True)
df = df.replace('Select', np.nan)


# In[8]:


df.isnull().sum()


# # checking percentage of null values

# In[9]:


round(100*(df.isnull().sum()/len(df.index)), 2)


# In[10]:


#Now we drop all those column who has more than 45% of missing values 
colum=df.columns

for i in colum:
    if((100*(df[i].isnull().sum()/len(df.index))) >= 45):
        df.drop(i, 1, inplace = True)


# In[11]:


round(100*(df.isnull().sum()/len(df.index)), 2)


# # Categorical Analysis 

# In[12]:


#Country
df['Country'].value_counts(dropna=False)


# # Droping the column

# In[13]:


colum_to_drop=['Country']
colum_to_drop.append('What matters most to you in choosing a course')
colum_to_drop.append('Search')
colum_to_drop.append('Magazine')
colum_to_drop.append('Newspaper Article')
colum_to_drop.append('X Education Forums')
colum_to_drop.append('Newspaper')
colum_to_drop.append('Digital Advertisement')
colum_to_drop.append('Through Recommendations')
colum_to_drop.append('Receive More Updates About Our Courses')
colum_to_drop.append('Update me on Supply Chain Content')
colum_to_drop.append('Get updates on DM Content')
colum_to_drop.append('I agree to pay the amount through cheque')
colum_to_drop.append('Do Not Call')
colum_to_drop


# In[14]:


df = df.drop(colum_to_drop,1)
df.info()


# In[15]:


round(100*(df.isnull().sum()/len(df.index)), 2)


# In[16]:


df.info()


# # Step 3
# # Handling the missing values

# In[17]:


#city
df['City'].value_counts(dropna=False)


# In[18]:


#Mumbai is the most common occurence among the non-missing values so we can impute all missing values with Mumbai
df['City'] = df['City'].replace(np.nan,'Mumbai')


# In[19]:


#Specialization
df['Specialization'].value_counts(dropna=False)


# In[20]:


#Here we will replace NaN values here with 'Not Specified'
df['Specialization'] = df['Specialization'].replace(np.nan, 'Not Specified')


# In[21]:


#combining Management Specializations 

df['Specialization'] = df['Specialization'].replace(['Finance Management','Human Resource Management','Marketing Management','Operations Management','IT Projects Management','Supply Chain Management','Healthcare Management','Hospitality Management','Retail Management'] ,'Management_Specializations')  


# In[22]:


df['Specialization'].value_counts(dropna=False)


# In[23]:


#What is your current occupation
df['What is your current occupation'].value_counts(dropna=False)


# In[24]:


#Here we impute the NaN values with unemployed
df['What is your current occupation'] = df['What is your current occupation'].replace(np.nan, 'Unemployed')


# In[25]:


df['What is your current occupation'].value_counts(dropna=False)


# In[26]:


round(100*(df.isnull().sum()/len(df.index)), 2)


# In[27]:


#tags
df['Tags'].value_counts(dropna=False)


# In[28]:


df['Tags'] = df['Tags'].replace(np.nan,'Not Specified')
df['Tags'] = df['Tags'].replace(['In confusion whether part time or DLP', 'in touch with EINS','Diploma holder (Not Eligible)','Approached upfront','Graduation in progress','number not provided', 'opp hangup','Still Thinking','Lost to Others','Shall take in the next coming month','Lateral student','Interested in Next batch','Recognition issue (DEC approval)','Want to take admission but has financial problems','University not recognized'], 'Other_Tags')
df['Tags'] = df['Tags'].replace(['switched off','Already a student','Not doing further education','invalid number','wrong number given','Interested  in full time MBA'] , 'Other_Tags')


# In[29]:


plt.figure(figsize=(15,5))
s1=sns.countplot(df['Tags'], hue=df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# In[30]:


#Lead Source
df['Lead Source'].value_counts(dropna=False)


# In[31]:


df['Lead Source'] = df['Lead Source'].replace(np.nan,'Others')
df['Lead Source'] = df['Lead Source'].replace('google','Google')
df['Lead Source'] = df['Lead Source'].replace('Facebook','Social Media')
df['Lead Source'] = df['Lead Source'].replace(['bing','Click2call','Press_Release','youtubechannel','welearnblog_Home','WeLearn','blog','Pay per Click Ads','testone','NC_EDM'] ,'Others')    


# In[32]:


plt.figure(figsize=(15,5))
s1=sns.countplot(df['Lead Source'], hue=df.Converted)
s1.set_xticklabels(s1.get_xticklabels(),rotation=90)
plt.show()


# Maximum number of leads are generated by Google and Direct traffic.

# In[33]:


round(100*(df.isnull().sum()/len(df.index)), 2)


# In[34]:


# Last Activity
df['Last Activity'].value_counts(dropna=False)


# In[35]:


df['Last Activity'] = df['Last Activity'].replace(np.nan,'Others')
df['Last Activity'] = df['Last Activity'].replace(['Unreachable','Unsubscribed','Had a Phone Conversation', 'Approached upfront','View in browser link Clicked',       'Email Marked Spam',                  
                                                        'Email Received','Resubscribed to emails','Visited Booth in Tradeshow'],'Others')


# In[36]:


df['Last Activity'].value_counts(dropna=False)


# In[37]:


round(100*(df.isnull().sum()/len(df.index)), 2)


# In[38]:


#Now we drop all those rows who has missing values as we can see dropped rows is less than 2%, it will not affect
df = df.dropna()


# In[39]:


round(100*(df.isnull().sum()/len(df.index)), 2)


# # Step 4
# # Numerical Analysis
# 
# correlations of numeric values

# In[40]:


plt.figure(figsize=(10,8))

# heatmap
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[41]:


#Total Visits
plt.figure(figsize=(8,6))
sns.boxplot(y=df['TotalVisits'])
plt.show()


# We can see the outliers here

# In[42]:


df['TotalVisits'].describe(percentiles=[0.05,.25, .5, .75, .90, .95, .99])


# In[43]:


Q3 = df.TotalVisits.quantile(0.99)
df = df[(df.TotalVisits <= Q3)]
Q1 = df.TotalVisits.quantile(0.01)
df = df[(df.TotalVisits >= Q1)]
sns.boxplot(y=df['TotalVisits'])
plt.show()


# In[44]:


#Page Views Per Visit
sns.boxplot(y=df['Page Views Per Visit'])
plt.show()


# In[45]:


#Outlier Treatment
Q3 = df['Page Views Per Visit'].quantile(0.99)
df = df[df['Page Views Per Visit'] <= Q3]
Q1 = df['Page Views Per Visit'].quantile(0.01)
df = df[df['Page Views Per Visit'] >= Q1]
sns.boxplot(y=df['Page Views Per Visit'])
plt.show()


# In[46]:


#Total Visits
sns.boxplot(y = 'TotalVisits', x = 'Converted', data = df)
plt.show()


# In[47]:


sns.boxplot(x=df.Converted, y=df['Total Time Spent on Website'])
plt.show()


# Nothng conclusive can be said on the basis of total visits but Leads spending more time on the website are more
# likely to be converted

# In[50]:


round(100*(df.isnull().sum()/len(df.index)),2)


# There are no missing values so we can build our model
# 

# # Dummy Variable
# 
# categorical columns

# In[51]:


cat_cols= df.select_dtypes(include=['object']).columns
cat_cols


# In[52]:


varlist =  ['A free copy of Mastering The Interview','Do Not Email']
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})
df[varlist] = df[varlist].apply(binary_map)


# In[53]:


dummy = pd.get_dummies(df[['Lead Origin','What is your current occupation','City']], drop_first=True)
df = pd.concat([df,dummy],1)


# In[54]:


dummy = pd.get_dummies(df['Lead Source'], prefix  = 'Lead Source')
dummy = dummy.drop(['Lead Source_Others'], 1)
df = pd.concat([df, dummy], axis = 1)


# In[55]:


dummy = pd.get_dummies(df['Last Activity'], prefix  = 'Last Activity')
dummy = dummy.drop(['Last Activity_Others'], 1)
df = pd.concat([df, dummy], axis = 1)


# In[56]:


dummy = pd.get_dummies(df['Tags'], prefix  = 'Tags')
dummy = dummy.drop(['Tags_Not Specified'], 1)
df = pd.concat([df, dummy], axis = 1)


# In[57]:


#dropping the original columns
df.drop(cat_cols,1,inplace = True)


# In[58]:


df.head()


# Now the data is ready to build a model 

# # Step 5
# # Train-Test Split & Logistic Regression Model Building

# In[59]:


from sklearn.model_selection import train_test_split
y = df['Converted']
y.head()
X=df.drop('Converted', axis=1)


# In[60]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# In[61]:


X_train.info()


# In[62]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols=X_train.select_dtypes(include=['float64', 'int64']).columns
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])


# In[63]:


X_train.head()


# # Model Building using Stats Model & RFE:

# In[64]:


import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


# In[65]:


# Running RFE with the output number of the variable equal to 15
logreg = LogisticRegression()

rfe = RFE(logreg, n_features_to_select=10)


# In[66]:


rfe = rfe.fit(X_train, y_train)


# In[67]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[68]:


col = X_train.columns[rfe.support_]
col


# In[69]:


X_train.columns[~rfe.support_]


# # Building model using statsmodel, for the detailed statistics

# In[70]:


X_train_sm = sm.add_constant(X_train[col])
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm1.fit()
res.summary()


# In[71]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[72]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[73]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[74]:


X_train_sm = sm.add_constant(X_train[col])
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[75]:


vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[76]:


y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[77]:


y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:10]


# In[78]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_prob':y_train_pred})
y_train_pred_final['Prospect ID'] = y_train.index
y_train_pred_final.head()


# In[79]:


y_train_pred_final['Predicted'] = y_train_pred_final.Converted_prob.map(lambda x: 1 if x > 0.5 else 0)

y_train_pred_final.head()


# In[80]:


from sklearn import metrics
 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.Predicted )
print(confusion)


# In[81]:


print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.Predicted))


# In[82]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[83]:


TP / float(TP+FN)


# In[84]:


TN / float(TN+FP)


# In[85]:


print(FP/ float(TN+FP))


# In[86]:


# positive predictive value 
print (TP / float(TP+FP))


# In[87]:


# Negative predictive value
print (TN / float(TN+ FN))


# PLOTTING ROC CURVE

# In[88]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[89]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Converted_prob, drop_intermediate = False )
draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# The ROC Curve should be a value close to 1. We are getting a good value of 0.95 indicating a good predictive model.

# # Finding Optimal Cutoff Point

# In[90]:


numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Converted_prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[91]:


cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# In[92]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])
plt.show()


# In[93]:


# From the curve above, 0.3 is the optimum point to take it as a cutoff probability.
y_train_pred_final['final_Predicted'] = y_train_pred_final.Converted_prob.map( lambda x: 1 if x > 0.3 else 0)
y_train_pred_final.head()


# In[94]:


y_train_pred_final['Lead_Score'] = y_train_pred_final.Converted_prob.map( lambda x: round(x*100))
y_train_pred_final[['Converted','Converted_prob','Prospect ID','final_Predicted','Lead_Score']].head()


# In[95]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[96]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion2


# In[97]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[98]:


TP / float(TP+FN)


# In[99]:


TN / float(TN+FP)


# 
# So as we can see above the model seems to be performing well. The ROC curve has a value of 0.95, which is very good. We have the following values for the Train Data:
# 
# Accuracy : 90.55%
# Sensitivity : 87.79%
# Specificity : 92.24%

# Some of the other Stats are derived below, indicating the False Positive Rate, Positive Predictive Value,Negative Predictive Values, Precision & Recall.

# In[100]:


# Calculate False Postive Rate - predicting conversion when customer does not have convert
print(FP/ float(TN+FP))


# In[101]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[102]:


# Negative predictive value
print (TN / float(TN+ FN))


# In[103]:


confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_Predicted )
confusion


# In[104]:


TP / TP + FP

confusion[1,1]/(confusion[0,1]+confusion[1,1])


# In[105]:


# Recall
TP / TP + FN

confusion[1,1]/(confusion[1,0]+confusion[1,1])


# In[106]:


from sklearn.metrics import precision_score, recall_score


# In[107]:


precision_score(y_train_pred_final.Converted , y_train_pred_final.final_Predicted)


# In[108]:


recall_score(y_train_pred_final.Converted, y_train_pred_final.final_Predicted)


# In[109]:


from sklearn.metrics import precision_recall_curve


# In[110]:


y_train_pred_final.Converted, y_train_pred_final.final_Predicted
p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_prob)


# In[111]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.show()


# In[112]:


num_cols=X_test.select_dtypes(include=['float64', 'int64']).columns

X_test[num_cols] = scaler.fit_transform(X_test[num_cols])

X_test.head()


# In[113]:


X_test = X_test[col]
X_test.head()


# In[114]:


X_test_sm = sm.add_constant(X_test)


# # PREDICTIONS ON TEST SET

# In[115]:


y_test_pred = res.predict(X_test_sm)


# In[116]:


y_test_pred[:10]


# In[117]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)
y_pred_1.head()


# In[118]:


y_test_df = pd.DataFrame(y_test)


# In[119]:


y_test_df['Prospect ID'] = y_test_df.index


# In[120]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[121]:


y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
y_pred_final.head()


# In[122]:


y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_prob'})


# In[123]:


y_pred_final.head()


# In[124]:


# Rearranging the columns
y_pred_final = y_pred_final[['Prospect ID','Converted','Converted_prob']]
y_pred_final['Lead_Score'] = y_pred_final.Converted_prob.map( lambda x: round(x*100))


# In[125]:


y_pred_final.head()


# In[126]:


y_pred_final['final_Predicted'] = y_pred_final.Converted_prob.map(lambda x: 1 if x > 0.3 else 0)


# In[127]:


y_pred_final.head()


# In[128]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# In[129]:


confusion2 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_Predicted )
confusion2


# In[130]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[131]:


TP / float(TP+FN)


# In[132]:


# Let us calculate specificity
TN / float(TN+FP)


# In[133]:


precision_score(y_pred_final.Converted , y_pred_final.final_Predicted)


# In[134]:


recall_score(y_pred_final.Converted, y_pred_final.final_Predicted)


# After running the model on the Test Data these are the figures we obtain:
# 
# Accuracy : 91.02%
# Sensitivity : 88.71%
# Specificity : 92.42%

# # Final Result 
# compare the values obtained for Train & Test:

# Train Data: Accuracy : 90.55% Sensitivity : 87.79% Specificity : 92.24%
#                 
# Test Data: Accuracy : 91.02% Sensitivity : 88.71% Specificity : 92.42%
#                 

# # Conclusion

# -Our Logistic Regression Model is decent and accurate enough
# 
# -X Education Company needs to focus on following key aspects to improve the overall conversion rate: a. Increase user engagement on their website since this helps in higher conversion b. Increase on sending SMS notifications since this helps in higher conversion c. Get TotalVisits increased by advertising etc. since this helps in higher conversion d. Improve the Olark Chat service since this is affecting the conversion negatively

# In[ ]:





# In[ ]:





# In[ ]:




