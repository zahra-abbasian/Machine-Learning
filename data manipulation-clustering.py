#!/usr/bin/env python
# coding: utf-8


"""
Project 1

Name: Zahra Abbasian

"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg

#VAT algorithm 
def VAT(R):
    """

    VAT algorithm adapted from matlab version:
    http://www.ece.mtu.edu/~thavens/code/VAT.m

    Args:
        R (n*n double): Dissimilarity data input
        R (n*D double): vector input (R is converted to sq. Euclidean distance)
    Returns:
        RV (n*n double): VAT-reordered dissimilarity data
        C (n int): Connection indexes of MST in [0,n)
        I (n int): Reordered indexes of R, the input data in [0,n)
    """
        
    R = np.array(R)
    N, M = R.shape
    if N != M:
        R = squareform(pdist(R))
        
    J = list(range(0, N))
    
    y = np.max(R, axis=0)
    i = np.argmax(R, axis=0)
    j = np.argmax(y)
    y = np.max(y)


    I = i[j]
    del J[I]

    y = np.min(R[I,J], axis=0)
    j = np.argmin(R[I,J], axis=0)
    
    I = [I, J[j]]
    J = [e for e in J if e != J[j]]
    
    C = [1,1]
    for r in range(2, N-1):   
        y = np.min(R[I,:][:,J], axis=0)
        i = np.argmin(R[I,:][:,J], axis=0)
        j = np.argmin(y)        
        y = np.min(y)      
        I.extend([J[j]])
        J = [e for e in J if e != J[j]]
        C.extend([i[j]])
    
    y = np.min(R[I,:][:,J], axis=0)
    i = np.argmin(R[I,:][:,J], axis=0)
    
    I.extend(J)
    C.extend(i)
    
    RI = list(range(N))
    for idx, val in enumerate(I):
        RI[val] = idx

    RV = R[I,:][:,I]
    
    return RV.tolist(), C, I


#Stage 1: Understand the dataset

#Question 1.1

#Reading the csv file and printing the number of rows and columns,etc
traffic = pd.read_csv('traffic.csv')
num_rows = len(traffic)
num_col = len(traffic.columns)

print('***')
print('Q1.1')
print('Number of traffic survey entries:', num_rows)
print('Number of attributes:', num_col)
print()
print(traffic.dtypes)
print('***')




# Question 1.2

#Replacing dashed values with Nan and removing them
traffic['maximum_speed'] = traffic['maximum_speed'].replace('-', np.nan)
traffic = traffic.dropna(axis=0, subset=['maximum_speed'])


print('***')
print('Q1.2')
print('Number of remaining traffic survey entries:', len(traffic))
print('***')



# Question 1.3

#Median value for attribute of 'vehicle_class_1'
med_veh_1 = traffic['vehicle_class_1'].median()

#Changing the type of attribute 'maximum_speed' from object to float
traffic["maximum_speed"] = pd.to_numeric(traffic["maximum_speed"], errors = 'coerce', downcast = 'float')

#maximum value for attribute 'maximum_speed'
max_speed = traffic['maximum_speed'].max()

print('***')
print('Q1.3')
print('Median value of vehicle_class_1: {:.1f}'.format(med_veh_1))
print('Highest value of maximum_speed: {:.1f}'.format(max_speed))
print('***')





#Stage 2: Data selection & manipulation

# Question 2.1

from pandas.io.json import json_normalize
import json

#Reading the json file and normalizing it to a dataframe
with open('roads.json') as f:
    roads_data = json.load(f)
    
roads = json_normalize(roads_data)

#Making a separate list of the desired elements (segment Ids and street types)
for elem in roads["data"]:
    roads_data = elem

seg_ID = []
street_type = []

for i in range(0,len(roads_data)):
    seg_ID.append(roads_data[i][9])
    street_type.append(roads_data[i][14])

#Changing the type of seg_ID from object to int
seg_ID = pd.to_numeric(seg_ID, errors = 'coerce', downcast = 'integer')

#Making a combined list and then a dataframe
zippedList =  list(zip(seg_ID, street_type))
street_df = pd.DataFrame(zippedList, columns = ['road_segment' , 'street_type']) 

#Merging into the main dataframe based on road_segment IDs
traffic = pd.merge(traffic, street_df, on = "road_segment", how = 'left')


print('***')
print("Q2.1")
print('The first three rows of traffic DataFrame with the attribute StrType are:')
print(traffic.head(3))
print('***')



# Question 2.2

#Creating a new column in the main dataframe by subtracting two attributes
traffic["max_speed_over_limit"] = traffic["maximum_speed"] - traffic["speed_limit"]

print('***')
print('Q2.2')
print('The first three rows of traffic DataFrame with the new max_speed_over_limit attribute are:')
print(traffic.head(3))
print('***')




# Question 2.3

#Creating a new dataframe based on street type
arterials = traffic.loc[traffic["street_type"] == "Arterial"]

#Grouping the new dataframe by road names and their maximum values
arter_road = arterials.groupby('road_name').max()

#Shows the three largest values and their groups
final_sort = arter_road["max_speed_over_limit"].nlargest(3)

print('***')
print('Q2.3')
print('Three Arterial roads with the highest maximum max_speed_over_limit:')
print(final_sort)
print('***')






#Stage 3: Visualisation and Clustering

#Plotting groups

get_ipython().run_line_magic('matplotlib', 'inline')
from numpy import arange

#Replacing dash values to Nan, then removing empty values from the dataframe
traffic['average_speed'] = traffic['average_speed'].replace('-', np.nan)
traffic = traffic.dropna(axis = 0, subset = ['average_speed'])

#Correction of the spelling of this suburb name
traffic['suburb'] = traffic['suburb'].replace('CARLTON', 'Carlton')

#Grouping by suburb names and their mean values
sub_name = traffic.groupby('suburb').mean()

average_speed = sub_name['average_speed']

sub_num = len(sub_name)
plt.bar(average_speed.index, average_speed, color = 'm')
plt.ylim(0,50)
plt.xlim(-1, sub_num)
plt.xticks(average_speed.index, rotation = 60)
plt.xlabel("Suburbs", fontsize = 15)
plt.ylabel("Average Speed (km/h)", fontsize = 15)
plt.title("Average speed in each suburb", fontsize = 20)

plt.show()




# Dimension reduction and visualisation


#A Tukey boxplot with outliers
plt.boxplot(traffic['vehicle_class_1'], meanline = True , showmeans = True , sym = 'c+')
plt.title("Distribution of vehicle_class_1 (with outliers)", fontsize = 20)
plt.xlabel("vehicle_class_1", fontsize = 15)
plt.ylabel("#per survey", fontsize = 15)
plt.show()

print()
print()

#A Tukey boxplot without outliers
plt.boxplot(traffic['vehicle_class_1'], meanline = True , showmeans = True, showfliers = False)
plt.title("Distribution of vehicle_class_1 (without outliers)", fontsize = 20)
plt.xlabel("vehicle_class_1", fontsize = 15)
plt.ylabel("#per survey", fontsize = 15)
plt.show()





# Clustering and visualisation

import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Reading the new file into a new dataframe
special_traffic = pd.read_csv('special_traffic.csv')

#Obtaining 1000 samples from the dataframe
sample_size = 1000
special_traffic = special_traffic.sample(sample_size)

#Putting the desired attributes into one list
#In this case I preferred this way over indexing method
features = ['maximum_speed','speed_limit','average_speed','vehicle_class_1','vehicle_class_2','vehicle_class_3',            'vehicle_class_4','vehicle_class_5','vehicle_class_6','vehicle_class_7','vehicle_class_8','vehicle_class_9',            'vehicle_class_10','vehicle_class_11', 'vehicle_class_12', 'vehicle_class_13', 'bike', 'motorcycle']

# Separating out the features
x = special_traffic.loc[:, features].values

# Separating out the rows by their street type
y = special_traffic.loc[:,['StrType']].values

#Standardising teh features
x = StandardScaler().fit_transform(x)

#Performing the PCA analysis and drawing the plot
pca = PCA(n_components = 2)

principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, special_traffic[['StrType']]], axis = 1)

plot = plt.figure(figsize = (10,10))
fig = plot.add_subplot(1,1,1) 
fig.set_xlabel('Principal Component 1', fontsize = 15)
fig.set_ylabel('Principal Component 2', fontsize = 15)
fig.set_title('Two-component PCA', fontsize = 20)
street_types = ['Arterial', 'Council Minor']
colors = ['b', 'g']
for elem, color in zip(street_types,colors):
    desired_points = finalDf['StrType'] == elem
    fig.scatter(finalDf.loc[desired_points, 'principal component 1']
               , finalDf.loc[desired_points, 'principal component 2']
               , c = color, s = 60)
fig.legend(street_types)
fig.grid()





# Question 3.2.c

#Creating a new dataframe based on desired attributes
traffics = special_traffic[features]

#Performing VAT analysis and drawing the plot
traffics_std = traffics
RV, R, I = VAT(traffics_std)
x = sns.heatmap(RV, cmap='viridis', xticklabels=False, yticklabels=False)
x.set(xlabel='Objects', ylabel='Objects')
x.set_title('VAT analysis on special_traffic dataframe', fontsize = 20)
plt.show()

print()
print()

# Question 3.2.c continued

#Performing VAT analysis using principal components
princ = principalDf[['principal component 1','principal component 2']]
princ_std = princ

RV, R, I = VAT(princ_std)
x = sns.heatmap(RV, cmap='viridis', xticklabels=False, yticklabels=False)
x.set(xlabel='Objects', ylabel='Objects')
x.set_title('VAT analysis based on Principal Components', fontsize = 20)
plt.show()




# Question 3.3.a

#Performing kmeans analysis and plotting the diagram
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(traffics)
    traffics["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.title("Eblow Curve for special_traffic", fontsize = 20)
plt.xlabel("Number of cluster", fontsize = 15)
plt.ylabel("SSE", fontsize = 15)
plt.show()




# Question 3.3.b

#Performing kmeans analysis for k=3 and plotting the clusters size
#In order to not mixing the data from above, k = i and there's a new list for SSE values
new_sse = {}
for i in range(1, 4):   
    kmeans = KMeans(n_clusters=i, max_iter=1000).fit(traffics)
    traffics["clusters"] = kmeans.labels_
    new_sse[i] = kmeans.inertia_ 
    
plt.bar(list(new_sse.keys()), list(new_sse.values()), color = 'violet')
plt.ylim(0,5e8)
plt.xlim(0, 4)
plt.xticks(list(new_sse.keys()), rotation=0)
plt.xlabel("Number of clusters (k)", fontsize = 15)
plt.ylabel("Cluster size", fontsize = 15)
plt.title("The size of clusters per k-mean value", fontsize = 15)

plt.show()



# Question 3.3.d

kmeans = KMeans(n_clusters=3).fit(principalDf)
centroids = kmeans.cluster_centers_

#Now plotting the clusters and centroids based on StrType
plot = plt.figure(figsize = (10,10))
fig = plot.add_subplot(1,1,1) 
fig.set_xlabel('Principal Component 1', fontsize = 15)
fig.set_ylabel('Principal Component 2', fontsize = 15)
fig.set_title('Kmeans clusters', fontsize = 20)
street_types = ['Arterial', 'Council Minor']
colors = ['b', 'g']
for elem, color in zip(street_types,colors):
    desired_points = finalDf['StrType'] == elem
    fig.scatter(finalDf.loc[desired_points, 'principal component 1']
               , finalDf.loc[desired_points, 'principal component 2']
               , c = color, s = 30)
fig.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100)   #Centroids shown as red dots
fig.legend(street_types)
fig.grid()




"""
Resources:

some parts of the codes here are taken from the following websites:
pandas.pydata.org
matplotlib.org
towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
geeksforgeeks.org
datacamp.com
stackoverflow.com
datasciencecentral.com
datatofish.com/k-means-clustering-python/

"""

