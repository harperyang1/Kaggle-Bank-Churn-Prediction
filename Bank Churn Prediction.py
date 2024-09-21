#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd

# for handling categorical variables
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# file system management
import os

#suppress warnings
import warnings
warnings.filterwarnings('ignore')

#matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

#print(os.listdir())
#print(os.getcwd())

###### read data 
train = pd.read_csv("/Users/yangzifan/Kaggle/Bank Churn Prediction/train.csv")
print('Training data shape:',train.shape)
train.head()

test = pd.read_csv("/Users/yangzifan/Kaggle/Bank Churn Prediction/test.csv")
print('Testing data shape:',test.shape)
test.head()

############################################################################################################ 1. exploratory data analysis
train['Exited'].value_counts()  #imbalanced class problem, have impact on choosing model performance metrics 

## very few missing values
def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val,mis_val_percent],axis=1)
    mis_val_table_rename = mis_val_table.rename(columns={0:"Missing Values",1:"% of Total Values"})
    # the table below only contains columns with missing values
    mis_val_table_rename = mis_val_table_rename[mis_val_table_rename.iloc[:,1]!=0].sort_values("% of Total Values",ascending=False).round(1)
    print('Your selected dataframe has '+str(df.shape[1])+' columns.\n'
         'There are '+str(mis_val_table_rename.shape[0])+' columns that have missing values.')
    return mis_val_table_rename  

missing_values = missing_values_table(train)
missing_values.head(14)

## check how many unique values in each field
train.dtypes.value_counts()
train.select_dtypes("object").apply(pd.Series.nunique,axis=0)
train['Geography'].unique()
train.select_dtypes("int64").apply(pd.Series.nunique,axis=0)
train.select_dtypes("float64").apply(pd.Series.nunique,axis=0)

## Tenure, NumofProducts have 11 and 4 categories respectively
cat_cols = ['Geography','Gender','Tenure','NumOfProducts','HasCrCard','IsActiveMember']
num_cols = ['CreditScore','Age','Balance','EstimatedSalary']
target = 'Exited'

## countplot of 0 and 1 for each category of each column in cat_cols 
fig = plt.figure(figsize=(14, len(cat_cols)*3))

for i, col in enumerate(cat_cols):
    
    plt.subplot(len(cat_cols)//2 + len(cat_cols) % 2, 2, i+1)
    sns.countplot(x=col, hue = target, data=train, palette=['#82b1fa','#237bfa']) #237bfa is dark blue
    plt.title(f"{col} Countplot by Target", fontweight = 'bold')
    plt.ylim(0, train[col].value_counts().max() + 10)
    
plt.tight_layout()
plt.show()

## count, max, mean, standard deviation, 25%, 50%, 75% of each num_col
train[num_cols].describe()

## boxplot for each num_col
plt.figure(figsize=(8, 6))
n_cols = 2
n_rows = (len(num_cols)+1)//n_cols # Calculate the number of rows needed
# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6))
# Flatten the axes array for easy iteration
axes = axes.flatten()
# Loop through each column and create a boxplot in its subplot
for i, col in enumerate(num_cols):
    sns.boxplot(y=train[col], ax=axes[i],palette=['#82b1fa'])
    axes[i].set_title(f'Boxplot of {col}')
# Remove any unused subplots (if the number of plots is odd)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.show()

## histogram for each num_col
plt.figure(figsize=(8, 6))
n_cols = 2
n_rows = (len(num_cols)+1)//n_cols # Calculate the number of rows needed
# Create subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10)) 
# if the second figure in figsize is too small, then the 
# Flatten the axes array for easy iteration
axes = axes.flatten()
for i, col in enumerate(num_cols):
    # Calculate mean and median
    mean_value = train[col].mean()
    median_value = train[col].median()
    # Add mean and median lines
    sns.histplot(x=train[col], ax=axes[i],palette=['#82b1fa'],bins=10)
    axes[i].set_title(f'Boxplot of {col}')
# Remove any unused subplots (if the number of plots is odd)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.show()


############################################################################################################ 2. feature engineering

## normalize numeric columns
for col in num_cols:
    sc = MinMaxScaler()
    train[col+"_scaled"] = sc.fit_transform(train[[col]])
    test[col+"_scaled"] = sc.fit_transform(test[[col]])
train.head(5)

# New features are added based on domain knowledge
# Perform the below transformation before getting dummies since the categorical variables will not exist after getting dummies
def add_new_features(df):
    df['AgeGroup'] = df['Age'] // 10 * 10
    df['IsSenior'] = df['Age'].apply(lambda x: 1 if x >= 60 else 0)
    df['QualityOfBalance'] = pd.cut(df['Balance'], bins=[-1,100,1000,10000,50000,300000], labels=['VeryLow', 'Low', 'Medium','High','Highest'])
    df['QualityOfBalance'].replace(['VeryLow', 'Low', 'Medium','High','Highest'],[0,1,2,3,4], inplace=True)
    df['Balance_to_Salary_Ratio'] = df['Balance'] / df['EstimatedSalary']
    df['CreditScoreTier'] = pd.cut(df['CreditScore'], bins=[0, 650, 750, 850], labels=['Low', 'Medium', 'High'])
    df['CreditScoreTier'].replace(['Low', 'Medium', 'High'],[0, 1, 2], inplace=True)
    df['IsActive_by_CreditCard'] = df['HasCrCard'] * df['IsActiveMember'] 
    df['Products_Per_Tenure'] =  df['Tenure'] / df['NumOfProducts']
    df['Customer_Loyalty_Status'] = df['Tenure'].apply(lambda x:0 if x < 2 else 1)
    return df

# In case CustomerID and surname have any impacts on the prediction 
def get_vectors(df_train,df_test,col_name):
    vectorizer = TfidfVectorizer(max_features=1000)
    vectors_train = vectorizer.fit_transform(df_train[col_name])
    vectors_test = vectorizer.transform(df_test[col_name])
    
    #Dimensionality Reduction Using SVD (Singular Value Decompostion) to capture the most important patterns or topics in data
    svd = TruncatedSVD(3)
    x_sv_train = svd.fit_transform(vectors_train)
    x_sv_test = svd.transform(vectors_test)

    # Convert to DataFrames
    tfidf_df_train = pd.DataFrame(x_sv_train)
    tfidf_df_test = pd.DataFrame(x_sv_test)

    # Naming columns in the new DataFrames
    cols = [(col_name + "_tfidf_" + str(f)) for f in tfidf_df_train.columns.to_list()]
    tfidf_df_train.columns = cols
    tfidf_df_test.columns = cols

    # Reset the index of the DataFrames before concatenation
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Concatenate transformed features with original data
    df_train = pd.concat([df_train, tfidf_df_train], axis="columns")
    df_test = pd.concat([df_test, tfidf_df_test], axis="columns")
    return df_train,df_test

train['CusId_Sur'] = train['CustomerId'].astype('str')+train['Surname']
test['CusId_Sur'] = test['CustomerId'].astype('str')+test['Surname']

# apply TF-IDF and SVD to CustomerID and Surname fields
train,test = get_vectors(train,test,'CusId_Sur') 

# generate new features based on domain knowledge
train = add_new_features(train)
test = add_new_features(test)

# transform categorical columns so they are ready to be used in logistic regression
train=pd.get_dummies(train,columns=cat_cols,drop_first=True)
#train.columns #len(train.columns)=30
test=pd.get_dummies(test,columns=cat_cols,drop_first=True)
#test.columns #len(test.columns)=29, the difference is Exited


correlations = train.corr()['Exited'].sort_values()
#print(correlations) # find out the features having the greatest linear relationship with Exited
# Extract the variables and show correlations
ext_data = train[['NumOfProducts_2','IsActiveMember_1.0','Gender_Male','Age_scaled','NumOfProducts_3','Geography_Germany',
                 'Balance_scaled','NumOfProducts_4']]
ext_data_corrs = ext_data.corr()
ext_data_corrs

plt.figure(figsize = (8, 6))
# Heatmap of correlations
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap')
# we can see there is a relatively strong linear correlation (abs value greater than 0.1) between numofproducts2 
# and age_scaled, numofproducts3, geography_germany, balance_scaled, and we are going to add polynomial 
# features for them to see if polynomial relationships exist 


# Extract the variables and show correlations
ext_data2 = train[['NumOfProducts_2','Age_scaled','NumOfProducts_3','Geography_Germany',
                 'Balance_scaled']]
ext_data_corrs2 = ext_data2.corr()
ext_data_corrs2

# Pairs plot, detect non-linear relationships
## Copy the data for plotting
plot_data = ext_data2.copy()
plot_data['TARGET'] = train['Exited']
# Drop na values and limit to first 100000 rows
plot_data = plot_data.dropna().loc[:100000, :]
# Function to calculate correlation coefficient between two columns
def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.2, .8), xycoords=ax.transAxes,
                size = 20)

# Create the pairgrid object
grid = sns.PairGrid(data = plot_data, diag_sharey=False,
                    hue = 'TARGET', 
                    vars = [x for x in list(plot_data.columns) if x != 'TARGET'])

# Upper is a scatter plot
grid.map_upper(plt.scatter, alpha = 0.2)
# Diagonal is a histogram
grid.map_diag(sns.kdeplot)
# Bottom is density plot
grid.map_lower(sns.kdeplot, cmap = plt.cm.OrRd_r);

plt.suptitle('Ext Source and Features Pairs Plot', size = 32, y = 1.05);

# make polynomial features
poly_features = train[['NumOfProducts_2','Age_scaled','NumOfProducts_3','Geography_Germany',
                 'Balance_scaled']]
poly_features_test = test[['NumOfProducts_2','Age_scaled','NumOfProducts_3','Geography_Germany',
                 'Balance_scaled']]
                                  
# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree = 3)
# Train the polynomial features
poly_transformer.fit(poly_features)
# Transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)

# get the feature names after polynomial transformations
poly_transformer.get_feature_names_out(input_features =['NumOfProducts_2','Age_scaled','NumOfProducts_3','Geography_Germany',
                 'Balance_scaled'])[:15]

#train.columns
#test.columns

# Put test features into dataframe
poly_features_train = pd.DataFrame(poly_features, 
                                  columns = poly_transformer.get_feature_names_out(['NumOfProducts_2',
                'Age_scaled','NumOfProducts_3','Geography_Germany','Balance_scaled']))
poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names_out(['NumOfProducts_2',
                'Age_scaled','NumOfProducts_3','Geography_Germany','Balance_scaled']))

# Concat polynomial features onto training and testing dataframe
train_poly = pd.concat([train, poly_features_train], axis=1)
test_poly = pd.concat([test, poly_features_test], axis=1)

# Alignment ensures that the columns are aligned (reordered if necessary) and both DataFrames have the same columns, in the same order
train_poly, test_poly = train_poly.align(test_poly, join = 'inner', axis = 1)
train_poly = pd.concat([train_poly,train['Exited']],axis=1)

# Print out the new shapes
#print('Training data with polynomial features shape: ', train_poly.shape)
#print('Testing data with polynomial features shape:  ', test_poly.shape)

train = train_poly
#train.columns
test = test_poly
#test.columns

##Selecting columns for use, remove those columns not usable for the logistic regression model later
# CustomerId and Surname already used for TF-IDF
# Num_cols already used in "add_new_features", avoid multi-collinearity
feat_cols=train.columns.drop(['id', 'CustomerId', 'Surname','CusId_Sur','Exited'])  
feat_cols=feat_cols.drop(num_cols)
#print(feat_cols)
#non_numeric_columns = train[feat_cols].select_dtypes(include=['object', 'category', 'string']).columns
#print("Non-numeric columns:")
#print(non_numeric_columns)
#train['CreditScoreTier'].head() # need to convert this column to numeric otherwise logistic regression will not work
#train['QualityOfBalance'].head()
train['CreditScoreTier'].astype('category').cat.codes
train['QualityOfBalance'].astype('category').cat.codes

############################################################################################################ 3. Model Fitting and Evaluation 

# 1. Set up Stratified K-Folds cross-validation
# cross validation reduces the risks of overfitting, maximizing data usage, and providing reliable performance estimates
# stratified k-fold cross validation ensures that each fold has a similar class distribution, giving a more reliable 
# generalizable performance on both dominant class and underrepresented folds
strat_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 2. Define the logistic regression model and hyperparameter grid
logreg = LogisticRegression()  
param_grid = {
    'C': np.logspace(-4, 4, 10),  # Regularization strength
    'max_iter': [100000], # a higher max_iter for convergence
    'solver': ['lbfgs'],
    'class':['balanced'] # the two classes are imbalanced
}

# 3. Set up GridSearchCV with Stratified K-Folds
grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=strat_kf, scoring='recall', n_jobs=-1) #false negative is more costly (need to 
# identify the customers with tendency to churn)
# we only pick one estimator for each grid search

# 4. Fit the model to the data (this performs the cross-validation and hyperparameter tuning)
grid_search.fit(train[feat_cols], train['Exited'])

# 5. Retrieve the best model and its parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f'Best Parameters: {best_params}')

# 6. Evaluate the model using cross-validation scores
cv_results = grid_search.cv_results_
print(f'Best Cross-Validated ROC AUC Score: {grid_search.best_score_:.4f}')

# 7. Make predictions and evaluate metrics on each fold if needed
y_pred = cross_val_predict(best_model, train[feat_cols], train['Exited'], cv=strat_kf)
y_pred_proba = cross_val_predict(best_model, train[feat_cols], train['Exited'], cv=strat_kf, method='predict_proba')[:, 1]

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(train['Exited'], y_pred)
print('Confusion Matrix:')
print(conf_matrix)

class_report = classification_report(train['Exited'], y_pred)
print('Classification Report:')
print(class_report)

# 8. Examine feature importance
# Extract feature importance (coefficients)
feature_importance = best_model.coef_[0]  # [0] for binary classification
# Get the feature names
features = train[feat_cols].columns

# Create a DataFrame to hold feature names and their corresponding importance
coef_df = pd.DataFrame({'Feature': features, 'Importance': feature_importance})
# Sort the DataFrame by importance
coef_df = coef_df.sort_values(by='Importance', ascending=False)

# Plotting the top 20 features having the greatest positive relationship with customer churn
plt.figure(figsize=(9, 6))
sns.barplot(x='Importance', y='Feature', data=coef_df[:20])
# Adjust y-axis label font size and make title more readable
plt.yticks(fontsize=12)  # Change the font size of y-axis labels
plt.xticks(fontsize=12)  # Change x-axis font size too
plt.title('Top 20 Important Features in Logistic Regression (with Positive Impacts)', fontsize=14)  # Title font size
plt.savefig('Top 20 Important Features in Logistic Regression (with Positive Impacts).png', bbox_inches='tight', dpi=300)
plt.show()

# Plotting the top 20 features having the greatest negative relationship with customer churn
plt.figure(figsize=(9, 6))
sns.barplot(x='Importance', y='Feature', data=coef_df[-20:])
# Adjust y-axis label font size and make title more readable
plt.yticks(fontsize=12)  # Change the font size of y-axis labels
plt.xticks(fontsize=12)  # Change x-axis font size too
plt.title('Top 20 Important Features in Logistic Regression (with Negative Impacts)', fontsize=14)  # Title font size
plt.savefig('Top 20 Important Features in Logistic Regression (with Negative Impacts).png', bbox_inches='tight', dpi=300)
plt.show()
