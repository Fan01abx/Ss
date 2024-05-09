

# In[2]:


# Step 1: Import libraries
get_ipython().system(' pip install -U -q PyDrive')


# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[4]:


iris_data = pd.read_csv ( 'iris.csv' )
iris_data


# In[6]:


# Step 3: Data preprocessing
# Split the dataset into features (X) and target variable (y)
X = iris_data.drop(columns=['Species'])
y = iris_data['Species']


# In[7]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Step 4: Choose Classification Techniques
# Initialize classifiers for Decision Trees, SVM, and KNN
dt_classifier = DecisionTreeClassifier()
svm_classifier = SVC()
knn_classifier = KNeighborsClassifier()


# In[9]:


# Step 5: Train and Evaluate Models
# Train each model using the training data
dt_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)


# In[11]:


# Make predictions on the testing data
dt_predictions = dt_classifier.predict(X_test)
svm_predictions = svm_classifier.predict(X_test)
knn_predictions = knn_classifier.predict(X_test)


# In[13]:


# Calculate accuracy for each model using accuracy_score()
dt_accuracy = accuracy_score(y_test, dt_predictions)
svm_accuracy = accuracy_score(y_test, svm_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)


# In[14]:


# Step 6: Compare Results
print("Decision Tree Accuracy:", dt_accuracy)
print("SVM Accuracy:", svm_accuracy)
print("KNN Accuracy:", knn_accuracy)


# In[15]:


import matplotlib.pyplot as plt

# Accuracy scores obtained
accuracies = [dt_accuracy, svm_accuracy, knn_accuracy]
classifiers = ['Decision Tree', 'SVM', 'KNN']

# Plotting the accuracies
plt.figure(figsize=(8, 6))
plt.bar(classifiers, accuracies, color=['blue', 'green', 'red'])
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Classifiers')
plt.ylim(0, 1)  # Set y-axis limit from 0 to 1 for accuracy percentage
plt.show()


# In[ ]:




