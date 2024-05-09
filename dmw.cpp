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



Assignment 2


# In[32]:


# a) find standard deviation,varience of every numerical attribute
std_dev = data.std()
varience = data.var()

print("Standard Deviation of each numerical attribute:")
print(std_dev)
print("\nVariance of each numerical attributes:")
print(varience)


# In[34]:


# b) Find covariance and perform Correlation analysis using Correlation coefficient
covariance_matrix = data.cov()
correlation_matrix = data.corr()
print("\nCovariance Matrix:\n", covariance_matrix)
print("\nCorrelation Matrix:\n", correlation_matrix)


# In[35]:


# c) How many independent features are present in the given dataset?
independent_features = np.linalg.matrix_rank(data.corr())
print("\nNumber of independent features:", independent_features)


# In[36]:


# d) Can we identify unwanted features?
# we can analyze correlation coefficients to identify features strongly correlated with each other.
# Features with high correlation might be candidates for removal if redundancy is present.
# Features with high correlation might be candidates for removal.

unwanted_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            colname = correlation_matrix.columns[i]
            unwanted_features.add(colname)

print("\nd) Unwanted features:", unwanted_features)


# In[37]:


# e) Perform data discretization using equi-frequency binning method on the age attribute
num_bins = 5  # Choose the number of bins as needed
data['age_bins'] = pd.qcut(data['Age'], q=num_bins, labels=False)


# In[38]:


# Visualize the equi-frequency bins graphically
plt.figure(figsize=(5, 3))
plt.hist(data['Age'], bins=num_bins, edgecolor='black', alpha=0.7)
plt.title('Equi-Frequency Binning for Age Attribute')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[53]:


# f) Normalize RestBP, chol, and MaxHR attributes using different normalization techniques
attributes_to_normalize = ['RestBP', 'chol', 'maxHR']

# Min-Max normalization
min_max_scaler = MinMaxScaler()
data_min_max_normalized = data.copy()
data_min_max_normalized[attributes_to_normalize] = min_max_scaler.fit_transform(data[attributes_to_normalize])


# In[50]:


# Z-score normalization
z_score_scaler = StandardScaler()
data_z_score_normalized = data.copy()
data_z_score_normalized[attributes_to_normalize] = z_score_scaler.fit_transform(data[attributes_to_normalize])


# In[51]:


# Decimal scaling normalization
decimal_scaling_factor = 10 ** (len(str(int(data[attributes_to_normalize].abs().max().max()))) - 1)
data_decimal_scaled = data.copy()
data_decimal_scaled[attributes_to_normalize] = data[attributes_to_normalize] / decimal_scaling_factor


# In[52]:


# Display the results
print("\nMin-Max Normalized DataFrame:\n", data_min_max_normalized.head())
print("\nZ-Score Normalized DataFrame:\n", data_z_score_normalized.head())
print("\nDecimal Scaled DataFrame:\n", data_decimal_scaled.head())


# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mlxtend')
get_ipython().system('pip install apyori graphviz')


# In[2]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
import graphviz


# In[3]:


#Create a sample dataset:
data = [['Milk', 'Bread', 'Butter'],
        ['Milk', 'Bread'],
        ['Milk', 'Eggs'],
        ['Bread', 'Eggs'],
        ['Milk', 'Bread', 'Eggs', 'Butter'],
        ['Tea', 'Bread', 'Eggs']]

df = pd.DataFrame(data, columns=['Item1', 'Item2', 'Item3', 'Item4'])
df


# In[4]:


#Convert the dataset to a transaction format:
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
df_encoded


# In[5]:


#Apply the Apriori algorithm:
frequent_itemsets_apriori = apriori(df_encoded, min_support=0.33, use_colnames=True)
frequent_itemsets_apriori


# In[6]:


#Apply the FP-growth algorithm:
frequent_itemsets_fpgrowth = fpgrowth(df_encoded, min_support=0.33, use_colnames=True)
frequent_itemsets_fpgrowth


# In[10]:


#Construction of FP-Tree
class Node:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}

def build_tree(data, min_support):
    header_table = {}
    for index, row in data.iterrows():
        for item in row:
            header_table[item] = header_table.get(item, 0) + 1

    for k in list(header_table):
        if header_table[k] < min_support:
            del header_table[k]

    frequent_items = list(header_table.keys())
    frequent_items.sort(key=lambda x: header_table[x], reverse=True)

    root = Node("Null", 1, None)

    for index, row in data.iterrows():
        ordered_items = [item for item in frequent_items if item in row]
        if ordered_items:
            insert_tree(ordered_items, root, header_table, 1)

    # Ensure 'Null' is in header_table
    if 'Null' not in header_table:
        header_table['Null'] = (0, None)

    return root, header_table

def insert_tree(items, node, header_table, count):
    if not items:
        return

    if items[0] in node.children:
        node.children[items[0]].count += count
    else:
        node.children[items[0]] = Node(items[0], count, node)

        if header_table[items[0]][1] is None:
            header_table[items[0]] = (header_table[items[0]][0], node.children[items[0]])
        else:
            update_header(header_table[items[0]][1], node.children[items[0]])

    if len(items) > 1:
        insert_tree(items[1:], node.children[items[0]], header_table, count)

def update_header(node_to_test, target_node):
    while node_to_test.nodeLink is not None:
        node_to_test = node_to_test.nodeLink
    node_to_test.nodeLink = target_node

# FP-tree construction
root, header_table = build_tree(df, min_support=2)

# Visualize the FP-tree
def visualize_tree(node, graph, parent_name, graph_name):
    if node is not None:
        graph.node(graph_name, f"{node.item} ({node.count})", shape="box")
        if parent_name is not None:
            graph.edge(parent_name, graph_name)
        for child_key, child_node in node.children.items():
            visualize_tree(child_node, graph, graph_name, f"{graph_name}_{child_key}")

# Create a graph using Graphviz
fp_tree_graph = graphviz.Digraph('FP_Tree', node_attr={'shape': 'box'}, graph_attr={'rankdir': 'TB'})
visualize_tree(root, fp_tree_graph, None, 'Root')

# Display the FP-tree visualization
fp_tree_graph.render(filename='fp_tree_visualization', format='png', cleanup=True)
fp_tree_graph


# In[11]:


# a) Find maximum frequent itemset
max_frequent_itemset_fp = frequent_itemsets_fpgrowth[frequent_itemsets_fpgrowth['support'] == frequent_itemsets_fpgrowth['support'].max()]
print("a) Maximum Frequent Itemset (FP-growth):\n", max_frequent_itemset_fp)

max_frequent_itemset_apriori = frequent_itemsets_apriori[frequent_itemsets_apriori['support'] == frequent_itemsets_apriori['support'].max()]
print("a) Maximum Frequent Itemset (Apriori):\n", max_frequent_itemset_apriori)


# In[12]:


# b) How many transactions does it contain?
num_transactions = len(df)
print("b) Number of transactions in the dataset:", num_transactions)


# In[13]:


# c) Simulate frequent pattern enumeration based on the FP-tree constructed.
def mine_patterns(node, prefix, header_table, min_support, patterns):
    if header_table[node.item][0] >= min_support:
        patterns.append(prefix + [node.item])

    for child_key, child_node in node.children.items():
        mine_patterns(child_node, prefix + [node.item], header_table, min_support, patterns)


patterns_apriori = list(frequent_itemsets_apriori['itemsets'])
print("c) Frequent Patterns Enumerated (Apriori):\n", patterns_apriori)
patterns_fp = []
mine_patterns(root, [], header_table, min_support=2, patterns=patterns_fp)
print("c) Frequent Patterns Enumerated (FP-growth):\n", patterns_fp)


# In[ ]:











