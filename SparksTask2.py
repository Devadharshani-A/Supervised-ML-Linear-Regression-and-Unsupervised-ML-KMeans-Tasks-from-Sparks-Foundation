#!/usr/bin/env python
# coding: utf-8

# # DEVADHARSHANI A
# # The sparks foundation-Task2
# ## Prediction using UnSupervised ML
# ## GRIP-@THE SPARKS FOUNDATION
# ### This is a simple machine learning task-K-means clustering, part of my tasks in the internship process at the sparks foundation.

# #### Importing libraries

# First lets import the Libraries

# In[182]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# #### Import dataset

# In[183]:


data=pd.read_csv(r"D:\job\internship\sparks\Iris (1).csv")


# #### Explore the dataset

# In[184]:


data.head()


# In[185]:


data.isnull().sum()


# In[186]:


data.describe()


# In[187]:


data.dtypes


# In[188]:


data.Species.unique()


# Here, the column id is not necessary, so lets drop it and lets convert the Species variable as a y variable, which we can use to compare at the last, if the cluster made by our clustering algorithm is correct or not.

# In[189]:


data.drop(columns=['Id'],inplace=True)


# In[190]:


x=data.iloc[:,[0,1,2,3]]
y=data.iloc[:,-1]


# In[191]:


x.head()


# In[192]:


y.head()


# In[193]:


# Convert y to DataFrame
y_df = pd.DataFrame(y, columns=['Species'])

# Perform mapping
y_df["Species"] = y_df["Species"].map({'Iris-setosa':1, 'Iris-versicolor':0, 'Iris-virginica':2})

# Convert y back to array
y = y_df['Species'].values


# In[194]:


print(y)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)print(y)Y=pd.DataFrame(y, columns=['Species'])Y.head()Y
# In[195]:


sns.pairplot(data=data,hue='Species')


# In[196]:


sns.boxplot(data=x)


# #### Finding optimum number of clusters

# Now, lets find the optimum number of clusters

# In[197]:


# Elbow chart to find optimum number of clusters

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
inertia_list = []
for i in range(1,10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(x)
    inertia_list.append(kmeans.inertia_)
plt.plot(range(1,10), inertia_list)
plt.title('Inertia_List vs No. of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('inertia_list')
plt.show()


# Here, we can find that 3 is the optimum number of clusters.

# In[219]:


kmeans=KMeans(n_clusters=3, init='k-means++',random_state=0)
pred=kmeans.fit_predict(x)
print(pred)


# In[220]:


x_array = np.array(x)


# #### Visualisation

# ##### Model's clustering

# In[224]:


# Visualising the clusters - On the first two columns
plt.scatter(x_array[pred == 0, 0], x_array[pred == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x_array[pred == 1, 0], x_array[pred == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x_array[pred == 2, 0], x_array[pred == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# ##### Original clusters

# In[225]:


# Visualising the clusters - On the first two columns
plt.scatter(x_array[y == 0, 0], x_array[y == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x_array[y == 1, 0], x_array[y == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x_array[y == 2, 0], x_array[y == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# We can note that there is no much difference in the clustering, and we can see that our model works well.

# #### Performance metrics

# In[228]:


from sklearn.metrics import accuracy_score,confusion_matrix
ac=accuracy_score(pred,y)
print("Accuracy score: ",ac)


# In[230]:


cm=confusion_matrix(pred,y)
cm


# I was successfully able to carry out prediction using unsupervised learning method-Kmeans clustering.

# #### Thank you
