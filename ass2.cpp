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


