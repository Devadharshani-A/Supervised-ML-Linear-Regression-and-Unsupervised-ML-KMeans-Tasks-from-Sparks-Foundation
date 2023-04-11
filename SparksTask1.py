#!/usr/bin/env python
# coding: utf-8

# # Author-Devadharshani A

# ## The sparks foundation-Task1
# ## Prediction using Supervised ML

# ### GRIP-@THE SPARKS FOUNDATION

# This is a simple machine learning task-simple linear regression, part of my tasks in the internship process at the sparks foundation.

# First step is to IMPORT the LIBRARIES

# In[1]:


#importing necessary libraries
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# next step is to import the dataset

# In[2]:


#importing dataset
url = "http://bit.ly/w-data"
data = pd.read_csv(url)


# In[3]:


#lets expore the data
data.head()


# In[4]:


data


# In[5]:


data.shape


# In[6]:


sns.scatterplot(x='Hours',y='Scores',data=data) 
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[7]:


data.dtypes


# In[8]:


data.isnull().sum()


# we can understand that, as the number of hours increases, the percentage also increases. Lets fit a linear regression into the data.

# Splitting the data for x and y, here, x will be the independent variable- Hours studied and y will be the dependent variable-Percentage scored

# In[9]:


x = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values 


# ### Train test split

# now we need to split the data for testing and training

# In[61]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)


# ### Fitting regression model 

# In[62]:


from sklearn.linear_model import LinearRegression  
reg = LinearRegression()
reg.fit(X_train.reshape(-1,1), y_train) 


# ### Prediction

# now we need to predict for xtest

# In[63]:


y_pred = reg.predict(X_test)


# In[64]:


y_pred


# ### Visualising the fitted line

# In[66]:


# Plotting the regression line
line = reg.coef_*x+reg.intercept_

# Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line,color='purple')
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ### Evaluation

# In[65]:


from sklearn.metrics import accuracy_score,mean_absolute_error


# In[67]:


ae=mean_absolute_error(y_pred,y_test)
print("Mean absolute error:  ",ae)


# In[68]:


acc=reg.score(X_test,y_pred)
print("Accuracy score: ",acc)


# In[69]:


print("Traning score: ",reg.score(X_train,y_train))


# now, lets test our model with a new data

# In[70]:


predval=reg.predict([[9.25]])
print("Number of hours=9.25")
print("Predicted percentage value for 9.25 hours: ",predval)


# ### Conclusion

# The model was successfully able to predict unseen data.
