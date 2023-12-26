#!/usr/bin/env python
# coding: utf-8

# In[113]:


#importing necessary libraries 
import pandas as pd 
import numpy as np  
from sklearn.cluster import KMeans  
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.pipeline import FeatureUnion
from math import pi
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv("C:/Users/kaurs/Downloads/E-commerce Customer Behavior - Sheet1.csv")


# In[5]:


print(df.head())


# In[63]:


numerical_features = df.select_dtypes(include=['int64','float64']).columns


# In[64]:


categorical_features =df.select_dtypes(include =['object']).columns


# In[66]:


numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown ='ignore'))
])


# In[67]:


preprocessor = ColumnTransformer(
    transformers = [
        ('num' , numerical_transformer, numerical_features),
        ('cat' ,categorical_transformer, categorical_features)
    ])


# In[85]:


kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)


# In[86]:


final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('kmeans', kmeans)
])


# In[87]:


labels = final_pipeline.fit_predict(df)


# In[99]:


cluster_centers = final_pipeline.named_steps['kmeans'].cluster_centers_


# In[101]:


plt.scatter(df['Total Spend'], df['Average Rating'], c=labels, cmap='viridis')
plt.title('Cluster Visualization')
plt.xlabel('Total Spend')
plt.ylabel('Average Rating')
plt.colorbar(label='Cluster')
plt.show()


# In[110]:


cluster_stats = df.groupby(labels).agg({'Total Spend': ['mean', 'std'], 'Items Purchased': 'median'})
print(cluster_stats)


# In[111]:


categorical_columns = ['Gender', 'City', 'Membership Type']  
for col in categorical_columns:
    cluster_counts = df.groupby([labels, col]).size().unstack()
    print(cluster_counts)


# In[112]:


sns.boxplot(x=labels, y='Total Spend', data=df)
plt.title('Total Spend Distribution by Cluster')
plt.show()



# In[129]:


cluster_stats = df.groupby(labels).agg({'Total Spend': ['mean', 'median', 'std'],
                                       'Items Purchased': ['mean', 'median', 'std']})
print(cluster_stats)


# In[130]:


plt.scatter(df['Total Spend'], df['Items Purchased'], c=labels, cmap='viridis')
plt.title('Total Spend vs. Items Purchased')
plt.xlabel('Total Spend')
plt.ylabel('Items Purchased')
plt.colorbar(label='Cluster')
plt.show()

sns.boxplot(x=labels, y='Age', data=df)
plt.title('Age Distribution by Cluster')
plt.show()


# In[134]:


from math import pi


additional_features = ['Age', 'Days Since Last Purchase', 'Satisfaction Level']


for cluster_label in range(len(cluster_centers)):
    values = cluster_centers[cluster_label][:len(additional_features)]
    num_features = len(additional_features)

    angles = [n / float(num_features) * 2 * pi for n in range(num_features)]

   
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles, additional_features, color='grey', size=10)

    if len(values) == len(angles):
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        ax.fill(angles, values, 'b', alpha=0.1)
        plt.title(f'Cluster {cluster_label+1} Radar Chart for Additional Features')
        plt.show()
    else:
        print("Values and angles have different lengths. Unable to plot radar chart.")


# In[136]:


def simulate_experiment(control_mean, variation_mean, control_std, variation_std, sample_size):
   
    control_group = np.random.normal(control_mean, control_std, sample_size)
    
 
    variation_group = np.random.normal(variation_mean, variation_std, sample_size)
    
    return control_group, variation_group

control_mean = 25  
variation_mean = 30  
control_std = 5 
variation_std = 5  
sample_size = 1000  


control_group, variation_group = simulate_experiment(control_mean, variation_mean, control_std, variation_std, sample_size)


from scipy import stats

t_stat, p_value = stats.ttest_ind(control_group, variation_group, equal_var=False)

alpha = 0.05  
if p_value < alpha:
    print("Statistically significant difference between groups. Variation group performs better.")
else:
    print("No statistically significant difference between groups. Further analysis may be needed.")

import matplotlib.pyplot as plt

plt.hist(control_group, alpha=0.5, label='Control Group')
plt.hist(variation_group, alpha=0.5, label='Variation Group')
plt.legend()
plt.title('Distribution of Metric in A/B Test')
plt.xlabel('Metric Value')
plt.ylabel('Frequency')
plt.show()


# In[ ]:




