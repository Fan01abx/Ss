Assignment 1 

# In[1]:


#importing pandas libraries
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#importing dataset
data = pd.read_csv('Heart.csv')
data


# In[3]:


# a) find missing values and replace with suitable attribute
missing_values = data.isnull().sum()
print("Missing Values:\n",missing_values)


# In[7]:


#we can choose a suitable alternative based on our dataset and context
#Here,we replace missing values with the mean of each column
data.fillna(data.mean(),inplace=True)


# In[8]:


missing_values = data.isnull().sum()
print("Missing Values:\n",missing_values)


# In[9]:


# b) remove inconsistancy
#You need to define the inconsistency based on your dataset. for example,removing duplicates.
data.drop_duplicates(inplace=True)


# In[33]:


# c) Boxplot analysis for each numerical attribute and find outliers
numerical_attributes = data.select_dtypes(include=['float64', 'int64']).columns
plt.figure(figsize=(15, 8))
for col in numerical_attributes:
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot for {col}')
    plt.show()


# In[23]:


data.boxplot()


# In[26]:


# d) Draw histogram for any two suitable attributes
plt.figure(figsize=(12,6))
plt.hist(data['Age'],bins=20,color='blue',alpha=0.7,label='Age')
plt.hist(data['Chol'],bins=20,color='orange',alpha=0.7,label='Chol')
plt.title("Histogram for Age and Chol Attributes")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()


# In[27]:


# e)find data type of each column
data_types = data.dtypes
print("Data Types:\n", data_types)


# In[29]:


# f)finding out zeros
zeros_count = (data == 0).sum()
print("Zeros Count:\n",zeros_count)


# In[30]:


# g) find mean age of patient
mean_age = data['Age'].mean()
print("Mean Age of Patients:",mean_age)


# In[31]:


# h) find shape of data
data_shape = data.shape
print("Shape of Data:", data_shape)


# In[ ]:










