#!/usr/bin/env python
# coding: utf-8

# # Introduction

# The purpose of this project will be to create a multivariable regression using the accepted loan dataset found in Kaggle. The source of the data is linked in the READ.DM file on github and contains 1,048,575 observations and 145 variables. Disclosure the source dataset found on Kaggle is extremely large at about 1.6 GB. For the purpose of this project, I have removed about 109 redundant variables to reduce computational slowdowns on python. 

# # Purpose

# Lending Club the company who this data is derived from has collected a million observations of people who got accepted for loans. Considering their data, the main goal for this project will be to determine the most significant variables that cause interest rates to change among borrowers who got accepted for loans. Therefor, I will be using a multivariable regression with **int_rate** (a variable that shows the percent of interest in a loan) , as the target variable. 

# # Importing the data

# In[1]:


import pandas as pd


# In[2]:


data=pd.read_csv('Loans.csv')


# # Data Exploration

# In[3]:


data.head()


# ------------

# Using data.info() we can see that there are 1,048,575 observations and 37 variables in the dataset note that there are both numerical and categorical variables in the data.

# In[4]:


data.info()


# --------

# Next, we want to check if null values exist in the data. Note that we want null values as a percentage value of all observations in a column. Considering the size of the dataset its better to know a percentage value  of null values rather than a vague null count.

# In[5]:


# Check for null values in each column
null_percentages = (data.isnull().sum() / len(data)) * 100

print(null_percentages)


# ---

# Now lets explore our categorical values and find how many unique categories exist in each categorical variable. The results here show that the variable **id** and **emp_title** have an abnormaly large amount of unique categories. 

# In[6]:


#lets check the number of unique categories in our categorical variables
for column in data.columns:
    if data[column].dtype == 'object':  # if the column is categorical
        num_unique_categories = data[column].nunique()
        print(f'{column}: {num_unique_categories} unique categories')


# # Data Cleaning

# Considering our previous findings, we need to drop both **id** and **emp_title** from the data. The reasoning behind this is because we’re going to create dummy variables for all the unique categories and it is impractical to create 237,261 dummy variables  for one categorical variable like **emp_title**.  

# In[7]:


#drops id and emp_title
data = data.drop(columns='id')
data = data.drop(columns='emp_title')


# ---

# While were at it, we might as well remove all null values from our categorical values. Considering that we had relatively low percentages of null values in our categorical variables. 

# In[8]:


# Removes rows/observations of null categorical variables
categorical_columns = data.select_dtypes(include=['object']).columns
data = data.dropna(subset=categorical_columns)


# ---

# Now that our categorical variables are free from null values and have workable amounts of unique categories, we can now convert our categorical variables into dummy variables. We will create a new variable called **data_dummies** and store the data there. 

# In[9]:


#creates dummy variables 
data_dummies = pd.get_dummies(data, drop_first=True)


# ---

# The following code now checks that our categorical variables have been converted correctly. All categorical variables will now be defined by the data type **uint8**.

# In[10]:


#Checks that the dummy variables changed to uint8
data_dummies.dtypes


# ---

# Now that our categorical variables are clean , we can now focus on cleaning the numerical variables. The following code imputes integer variables with a columns mean.Note that were still storing the imputation in the data_dummies variable. 

# In[11]:


#Imputes the mean for all integer variables with null values
numerical_columns = data_dummies.select_dtypes(include=['int64', 'float64']).columns
for column in numerical_columns:
    data_dummies[column].fillna(data_dummies[column].mean(), inplace=True)


# ---

# Lets now check that our dataset *data_dummies* is free from null values. 

# In[12]:


# Lets check again for Null values
null_percentages = (data_dummies.isnull().sum() / len(data)) * 100

print(null_percentages)


# # Multicollinearity

# Before starting our regression model, it is important to check for multicollinearity. Multicollinearity refers to a situation in which two or more explanatory variables in a regression model are highly linearly related. For that I will be using Variance Inflation Factor (VIF) note that if VIF is 1 then the variables are not correlated, similarly a value greater than 1 indicates that the variables are correlated. If VIF is very high (greater than 5 or 10), then the variables are highly correlated and we must consider dropping them one at a time. Usually this means we must drop the variable with the highest VIF value. 

# In[13]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

# Select only the numeric columns
numeric_columns = data_dummies.select_dtypes(include=['int64', 'float64'])

# Add a constant column for the VIF calculation
numeric_columns['constant'] = 1

# Calculate the VIF for each column
vif = pd.DataFrame()
vif["Variable"] = numeric_columns.columns
vif["VIF"] = [variance_inflation_factor(numeric_columns.values, i) for i in range(numeric_columns.shape[1])]

print(vif)


# ---

# Since the variable **loan_amnt** had the highest variance we must drop it first and then check for VIF again.

# In[14]:


# Drop the 'loan_amnt' variable
data_dummies = data_dummies.drop(columns='loan_amnt')

# Recalculate the VIFs
numeric_columns = data_dummies.select_dtypes(include=['int64', 'float64'])
numeric_columns['constant'] = 1
vif = pd.DataFrame()
vif["Variable"] = numeric_columns.columns
vif["VIF"] = [variance_inflation_factor(numeric_columns.values, i) for i in range(numeric_columns.shape[1])]
print(vif)


# ___

# Now we must drop the variable **pub_rec** and check that we have less than 5 VIF in our model.

# In[15]:


# Drop the 'pub_rec' variable
data_dummies = data_dummies.drop(columns='pub_rec')

# Recalculate the VIFs
numeric_columns = data_dummies.select_dtypes(include=['int64', 'float64'])
numeric_columns['constant'] = 1
vif = pd.DataFrame()
vif["Variable"] = numeric_columns.columns
vif["VIF"] = [variance_inflation_factor(numeric_columns.values, i) for i in range(numeric_columns.shape[1])]
print(vif)


# # Feature Scaling

# Since our VIF checked out okay we can now focus on feature scaling. Feature scaling is important because it can help the regression algorithm to converge more quickly and can also prevent certain features from dominating simply because they have a larger scale. The two most common forms of scaling are standardization and normalization. In order to know which form to use we must first check if our data has a Gaussian distribution if it does, we can use Standardization if not we must use Normalization.

# In[16]:


import matplotlib.pyplot as plt

# Lets check if our target variable is normally distributed
plt.hist(data_dummies['int_rate'], bins=30)
plt.show()


# ---

# 
# The histogram tells us that the data is not normally distributed. Therefor me must use normalization to scale our data. Normalization (or min-max scaling) will rescale our numerical variables into features that will range from 0 to 1.
# 
# Note that normalization must only occur with our numerical variables, therefor will first create a scaler object for normalization (*MinMaxScaler()*) and then we will isolate the numerical variables into a new variable called **numeric_columns**. After the numeric variables are isolated, we will then scale them using our scaler object. In the scaling process the scaled numeric variables will be assigned a new name **data_scaled**. Finally, we will update the **data_scaled** variable by converting the scaled numeric variables back into a dataframe.

# In[17]:


from sklearn.preprocessing import MinMaxScaler

# Create a scaler object
scaler = MinMaxScaler()

# Select only the numeric columns
numeric_columns = data_dummies.select_dtypes(include=['int64', 'float64'])


# Fit the scaler and transform the data
data_scaled = scaler.fit_transform(numeric_columns)

# Convert back to a dataframe
data_scaled = pd.DataFrame(data_scaled, columns=numeric_columns.columns)


# ---

# Having scaled the numeric variables we know must isolate our categorical variables into a new variable called **dummy_colums**

# In[ ]:


# Get the dummy variables from the original DataFrame
dummy_columns = data_dummies.select_dtypes(include=['uint8'])


# ---

# Before we can combine our newly scaled numeric variables with our categorical data, we must first reset the indices. The reason for this is because, when we use pd.concat() to merge our scaled numeric and categorical variables , pandas will try to align the data based on indices. If the indices in **data_scaled** and **dummy_columns** don't match, we will get NaN values in the resulting DataFrame.

# In[ ]:


# Reset the indices of both DataFrames
data_scaled = data_scaled.reset_index(drop=True)
dummy_columns = dummy_columns.reset_index(drop=True)

# Combine the scaled DataFrame and the dummy variables
data_scaled = pd.concat([data_scaled, dummy_columns], axis=1)


# ---

# # Linear Regression

# Now that our data is clean, scaled, and has been checked for variance, we can now start building the multivariable regression model. The following lines of code will split our data set into a training set of 85% and test set 15% and will store the values into 4 varialbles **X_train**, **X_test**, **y_train**, **y_test**.

# In[18]:


from sklearn.model_selection import train_test_split

# Lets train the data 
X = data_scaled.drop('int_rate', axis=1)
y = data_scaled['int_rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


# Next we will create a model and fit our training data

# In[19]:


from sklearn.linear_model import LinearRegression

# Create a model object
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)


# Finally we will run the model 

# In[20]:


import statsmodels.api as sm

# Separate the predictors (X) and the target variable (y)
X = data_scaled.drop('int_rate', axis=1)
y = data_scaled['int_rate']

# Add a constant to the predictors
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X)
results = model.fit()

# Print out the statistics
print(results.summary())


# # Interpreting the model

# The following are some basic interpretations that can be made of the model. Similarly, all the variables displayed by the regression result are deemed to be significant at predicting interest rate. 
# 
# **Adjsuted R-squared**:approximately 91.7% of the variability in interest rate can be explained by the predictor variables shown in the regression results. 
# 
# **Numeric Coefficients** : Since our data was normalized we need to interpret the variables differently, below I attached examples on how you would interpret the numeric variable *term* if it was standaridized, normalized, and not scaled . 
# 
# *Term (Not Scaled)*:For each additional month in the term of the loan, the interest rate increases by 0.0083 percentage points on average, assuming all other factors remain constant.
# 
# *term (Standardized)*: for each increase of one standard deviation in the term of the loan, the interest rate increases by 0.0083 standard deviations on average, assuming all other factors remain constant.
# 
# *term (Normalized)*:if the length of the loan term goes from the shortest possible term to the longest possible term, the model predicts that the interest rate will increase by an amount equal to 0.83% of the total possible range of interest rates.
# 
# **Categorical Coefficients**: For categorical variables we must compare the coefficient with the zero group. In this case for hardship_flag the zero group is hardship_flag_N.
# 
# *Hardship_flag_Y (Not Scaled)*:the presence of economic hardship (hardship_flag_Y'= 1) is associated with an increase in interest rate by 0.0211 units compared to the absence of economic hardship (hardship_flag_Y' = 0).

# In[ ]:




