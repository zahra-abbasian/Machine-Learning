#!/usr/bin/env python
# coding: utf-8




"""
COMP20008 Project 2

Name: Zahra Abbasian Korbekandi

"""





# Import libraries

import pandas as pd

from functools import partial

from fuzzywuzzy import process

from fuzzywuzzy import fuzz

import numpy as np

import matplotlib.pyplot as plt




# Part 1 (Data Linkage)


# Naive Data Linkage Without Blocking

# load in data from two files
amazon_small = pd.read_csv('amazon_small.csv')
google_small = pd.read_csv('google_small.csv')


#combined= amazon_small.merge(google_small, left_on='title', right_on='name', how='outer', 
                        #suffixes=["_amazon","_google"]) #since they are the same name, we need a suffix
    
# A function that iterates through one dataframe and matches the strings from the other dataframe                    
def match_name(name, list_names, min_score=0):
    max_score = -1           # a flag for a no-match
    max_name = ""            # Return empty for a no-match
    
    for name2 in list_names:        # Iterate over names in the other dataframe
        score = fuzz.token_set_ratio(name, name2)
        if (score > min_score) & (score > max_score):    # check if above the threshhold
            max_name = name2
            max_score = score
    return (max_name, max_score)



# create a list for dicts to store the matches
dict_list = []

#now iteration in the main program using the defined function above
for name in google_small.name:      
    match = match_name(name, amazon_small.title, 75)   #the threshhold is set here, as 75%
    dict_ = {}
    dict_.update({"Google_name" : name})
    dict_.update({"match_name" : match[0]})
    dict_.update({"score" : match[1]})
    dict_list.append(dict_)
    
matched_table = pd.DataFrame(dict_list)

matched_table





# merging the matched table with the origin tables and keeping only the ID coulumns
final_match = matched_table.merge(amazon_small, how='inner', left_on='match_name', right_on='title')
final_match = final_match.merge(google_small, how = 'inner', left_on = 'Google_name', right_on = 'name')
final_match = final_match[['idAmazon', 'idGoogleBase']]
final_match





# Load in the truth_small file and compare the results by merging the dataframes

truth_small = pd.read_csv('amazon_google_truth_small.csv')

truth_small2 = truth_small.merge(final_match, how = 'left', on = 'idGoogleBase')

#compare with truth list and create a new column that stores either 1 for a match or 0 for a non-match
truth_small2['idMatch?'] = np.where(truth_small2.idAmazon_x == truth_small2.idAmazon_y, 1 , 0) 

true_pos = truth_small2['idMatch?'].sum()    #number of matches (sum of numbers 1) as true positive
false_pos = len(final_match.index) - true_pos     #the difference between found matches and true positives
false_neg = truth_small2['idAmazon_y'].isna().sum()   #when a match is in the truth file but not found on the matched dataframe


recall = true_pos/(true_pos + false_neg)

precision = true_pos/(true_pos + false_pos)

print("Recall =", recall)
print("Precision =", precision)




    

# Blocking For Efficient Data Linkage

from math import*
import nltk


#A function that calculates the jaccard similarity which returns a -1 flag to avoid 'division by zero' error
def jaccard_similarity(x,y):

    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    if (union_cardinality == 0):
        return -1              #flag to avoid division by zero        
    else:
        return intersection_cardinality/float(union_cardinality)

# Same matching function from above with a little difference

def match_name(name, list_names, min_score=0):
    max_score = -1
    max_name = ""
    name.replace(" ", "")   #remove the whitespace between words
    name_chars = set(nltk.ngrams(name, n=3))   #create n-grams (in this case = 3)
    
    for name2 in list_names:
        name2.replace(" ", "")      #remove whitespace from the other dataframe names
        name2_chars = set(nltk.ngrams(name2, n=3))    #again create n-grams for the other dataframe

        score = jaccard_similarity(name_chars, name2_chars)
        
        if (score > min_score) & (score > max_score):     #check for being above threshhold
            max_name = name2
            max_score = score
    return (max_name, max_score)


#load in the two files
df_google = pd.read_csv('google.csv')

df_amazon = pd.read_csv('amazon.csv')

#main program same as above for the fuzzy linkage
dict_list = []

for name in df_amazon.title:
    
    match = match_name(name, df_google.name, 0.20)   #thresshold for similarity is set here
    
    dict_ = {}
    dict_.update({"amazon_name" : name})
    dict_.update({"google_name" : match[0]})
    dict_.update({"score" : match[1]})
    dict_list.append(dict_)
    
matched_table = pd.DataFrame(dict_list)
matched_table





# Merging the two dataframes and extracting only ID columns to compare with the truth file later

matched_table['google_name'].replace('', np.nan, inplace=True)
matched_table.dropna(subset=['google_name'], inplace = True)

matched_final = matched_table.merge(df_google, how = 'inner', left_on = 'google_name', right_on = 'name')
matched_final = matched_final.merge(df_amazon, how='inner', left_on='amazon_name', right_on='title')
matched_final = matched_final[['idAmazon', 'id']]

#load in the truth file
truth = pd.read_csv('amazon_google_truth.csv')

truth2 = truth.merge(matched_final, how = 'left', left_on = 'idGoogleBase', right_on = 'id')

#as above, compares the found matches with the truth lists and creates a new column, sets 1 for a match, 0 for non-match
truth2['idMatch?'] = np.where(truth2.idAmazon_x == truth2.idAmazon_y, 1 , 0)  


true_pos = truth2['idMatch?'].sum()            #sum of number 1s as true positives
false_pos = len(matched_final.index) - true_pos      #the difference between the len of truth and the resulted matches
false_neg = truth2['idAmazon_y'].isna().sum()        #false negatives are the empty fields in front of truth list

#the number of all possible pairs which is m*n (len of amazon by len of google lists)
n = len(df_amazon.index)*len(df_google.index)  

recall = true_pos/(true_pos + false_neg)

reduction_ratio = 1 - (true_pos + false_pos)/n

print("Recal =", recall)
print("Reduction Ratio =", reduction_ratio)
display(truth2)






# Part 2 - Classification



# Pre-processing

# Impute missing values:


#first, load in the yeast data
all_yeast = pd.read_csv('all_yeast.csv')
all_yeast2 = all_yeast      #duplicating the data to use later





# show the min, max, mean, median(50%) of the loaded data
all_yeast2.describe()



# Filling NaN values with median values of each column

all_yeast.fillna(all_yeast.median()['mcg':'nuc'], inplace = True)
all_yeast.describe()




# Filling NaN values with mean values of each column

all_yeast2.fillna(all_yeast.mean()['mcg':'nuc'], inplace = True)
all_yeast2.describe()




# Scale the features

from sklearn import preprocessing

#just keep the numeric data from the (median imputed) yeast dataframe
yeast_data = all_yeast[['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox' , 'vac' , 'nuc']].astype(float)

yeast_data2 = yeast_data   #duplicating to use later 

col_names = yeast_data.columns  # list of column names



#Standardization

scaler = preprocessing.StandardScaler()
stand_yeast = scaler.fit_transform(yeast_data2)
stand_yeast = pd.DataFrame(stand_yeast, columns = col_names)
stand_yeast.describe()



# mean centering

yeast_centered = preprocessing.scale(yeast_data, with_mean= True, with_std= False)
yeast_centered = pd.DataFrame(yeast_centered, columns = col_names)
yeast_centered.describe()




# Comparing classification algorithms


from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

#load in the numeric data
yeast_data = all_yeast[['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox' , 'vac' , 'nuc']].astype(float)
#separate the class column
class_label = all_yeast['Class']   

#form the training and testing lists
X_train, X_test, y_train, y_test = train_test_split(yeast_data, class_label, train_size=2/3, test_size=1/3, random_state=42)

#preprocessing of the data (median imputed, mean centered)
X_train = preprocessing.scale(X_train, with_mean= True, with_std= False)
X_test = preprocessing.scale(X_test, with_mean= True, with_std= False)


# k-NN algorithm (k = 5)
knn = neighbors.KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train, y_train)
#predicting
y_pred = knn.predict(X_test)
#evaluation of the algorithm
print("Accuracy is:", accuracy_score(y_test, y_pred))



# k-NN algorithm (k = 10)

#form the training and testing lists
X_train, X_test, y_train, y_test = train_test_split(yeast_data, class_label, train_size=2/3, test_size=1/3, random_state=42)

#preprocessing of the data (median imputed, mean centered)
X_train = preprocessing.scale(X_train, with_mean= True, with_std= False)
X_test = preprocessing.scale(X_test, with_mean= True, with_std= False)


knn = neighbors.KNeighborsClassifier(n_neighbors = 10)  #k=10
knn.fit(X_train, y_train)
#predicting
y_pred = knn.predict(X_test)
#evaluation of the algorithm
print("Accuracy is:", accuracy_score(y_test, y_pred))




# Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier 

dtree = DecisionTreeClassifier(criterion="entropy",random_state=1, max_depth=3)
dtree.fit(X_train, y_train)
#predicting
y_pred = dtree.predict(X_test)
#evaluation of the algorithm
print("Accuracy is:", accuracy_score(y_test, y_pred))





# Feature Engineering


# 1) Interaction term pairs

# The step by step method used for this part is first adding new columns 
# in the original yeast dataframe (after imputation), having values of f1*f2 or f1/f2
# Then the new dataframe is mean-centered and classified using k-NN algorithm.


all_yeast = pd.read_csv('all_yeast.csv')
all_yeast.fillna(all_yeast.median()['mcg':'nuc'], inplace = True)
# adding new columns


all_yeast['mcg/gvh'] = all_yeast['mcg']/all_yeast['gvh']
all_yeast['mcg*gvh'] = all_yeast['mcg']*all_yeast['gvh']
all_yeast['alm*vac'] = all_yeast['alm']*all_yeast['vac']
all_yeast['mit*nuc'] = all_yeast['mit']*all_yeast['nuc']
all_yeast['alm/gvh'] = all_yeast['alm']/all_yeast['gvh']



#Rearranging the order of the columns to make their selection easier
all_yeast = all_yeast[['Sample','mcg','gvh','alm','mit','vac','nuc',                     'mcg*gvh','alm*vac','mit*nuc', 'mcg/gvh', 'alm/gvh', 'erl', 'pox','Class']]


yeast_data = all_yeast[['mcg','gvh', 'alm', 'mit', 'vac', 'nuc','mcg*gvh','alm*vac','mit*nuc', 'mcg/gvh', 'alm/gvh']].astype(float)


class_label = all_yeast['Class']

X_train, X_test, y_train, y_test = train_test_split(yeast_data, class_label, train_size=2/3, test_size=1/3, random_state=42)


X_train = preprocessing.scale(X_train, with_mean= True, with_std= False)
X_test = preprocessing.scale(X_test, with_mean= True, with_std= False)


knn = neighbors.KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy is:", accuracy_score(y_test, y_pred))





# selecting different features #1


all_yeast = pd.read_csv('all_yeast.csv')
all_yeast.fillna(all_yeast.median()['mcg':'nuc'], inplace = True)

#Creating new features and storing them as new columns
all_yeast['mcg/gvh'] = all_yeast['mcg']/all_yeast['gvh']
all_yeast['mcg*gvh'] = all_yeast['mcg']*all_yeast['gvh']
all_yeast['alm*vac'] = all_yeast['alm']*all_yeast['vac']
all_yeast['mit*nuc'] = all_yeast['mit']*all_yeast['nuc']




#Rearranging the order of the columns to make their selection easier
all_yeast = all_yeast[['Sample','mcg','gvh','alm','mit','vac','nuc',                     'mcg*gvh','alm*vac','mit*nuc', 'mcg/gvh', 'erl', 'pox','Class']]

#Selecting features
yeast_data = all_yeast[['mcg','gvh', 'alm', 'mit', 'vac', 'nuc','mcg*gvh','alm*vac','mit*nuc', 'mcg/gvh']].astype(float)


class_label = all_yeast['Class']

X_train, X_test, y_train, y_test = train_test_split(yeast_data, class_label, train_size=2/3, test_size=1/3, random_state=42)

#mean centering
X_train = preprocessing.scale(X_train, with_mean= True, with_std= False)
X_test = preprocessing.scale(X_test, with_mean= True, with_std= False)


knn = neighbors.KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy is:", accuracy_score(y_test, y_pred))




# Testing different features #2

all_yeast = pd.read_csv('all_yeast.csv')
all_yeast.fillna(all_yeast.median()['mcg':'nuc'], inplace = True)

#Creating new features and storing them as new columns
all_yeast['mcg/gvh'] = all_yeast['mcg']/all_yeast['gvh']
all_yeast['mcg*gvh'] = all_yeast['mcg']*all_yeast['gvh']
all_yeast['alm*vac'] = all_yeast['alm']*all_yeast['vac']
all_yeast['mit*nuc'] = all_yeast['mit']*all_yeast['nuc']


#Rearranging the order of the columns to make their selection easier
all_yeast = all_yeast[['Sample','mcg','gvh','alm','mit','vac','nuc',                     'mcg*gvh','alm*vac','mit*nuc', 'mcg/gvh', 'erl', 'pox','Class']]

#Selecting features
yeast_data = all_yeast[['mcg','gvh', 'alm', 'mit', 'vac', 'nuc','mcg*gvh','alm*vac','mit*nuc', 'mcg/gvh']].astype(float)


class_label = all_yeast['Class']

X_train, X_test, y_train, y_test = train_test_split(yeast_data, class_label, train_size=2/3, test_size=1/3, random_state=42)

#mean centering
X_train = preprocessing.scale(X_train, with_mean= True, with_std= False)
X_test = preprocessing.scale(X_test, with_mean= True, with_std= False)


knn = neighbors.KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy is:", accuracy_score(y_test, y_pred))




# Testing different features #3  (The highest accuracy so far)

all_yeast = pd.read_csv('all_yeast.csv')
all_yeast.fillna(all_yeast.median()['mcg':'nuc'], inplace = True)

#Creating new features and storing them as new columns
all_yeast['mcg/gvh'] = all_yeast['mcg']/all_yeast['gvh']
all_yeast['alm*vac'] = all_yeast['alm']*all_yeast['vac']
all_yeast['mit*nuc'] = all_yeast['mit']*all_yeast['nuc']




#Rearranging the order of the columns to make their selection easier
all_yeast = all_yeast[['Sample','mcg','gvh','alm','mit','vac','nuc',                     'alm*vac','mit*nuc', 'mcg/gvh', 'erl', 'pox','Class']]

#Selecting features
yeast_data = all_yeast[['mcg','gvh', 'alm', 'mit', 'vac', 'nuc','alm*vac','mit*nuc', 'mcg/gvh']].astype(float)


class_label = all_yeast['Class']

X_train, X_test, y_train, y_test = train_test_split(yeast_data, class_label, train_size=2/3, test_size=1/3, random_state=42)

#mean centering
X_train = preprocessing.scale(X_train, with_mean= True, with_std= False)
X_test = preprocessing.scale(X_test, with_mean= True, with_std= False)


knn = neighbors.KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy is:", accuracy_score(y_test, y_pred))





from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2

all_yeast = pd.read_csv('all_yeast.csv')
all_yeast.fillna(all_yeast.median()['mcg':'nuc'], inplace = True)


all_yeast['mcg/gvh'] = all_yeast['mcg']/all_yeast['gvh']
all_yeast['mcg*gvh'] = all_yeast['mcg']*all_yeast['gvh']
all_yeast['alm*vac'] = all_yeast['alm']*all_yeast['vac']
all_yeast['mit*nuc'] = all_yeast['mit']*all_yeast['nuc']
all_yeast['alm/gvh'] = all_yeast['alm']/all_yeast['gvh']
all_yeast['mcg/alm'] = all_yeast['mcg']/all_yeast['alm']
all_yeast['mcg*alm'] = all_yeast['mcg']*all_yeast['alm']
all_yeast['mcg*mit'] = all_yeast['mcg']*all_yeast['mit']
all_yeast['mcg*vac'] = all_yeast['mcg']*all_yeast['vac']
all_yeast['mcg*nuc'] = all_yeast['mcg']*all_yeast['nuc']
all_yeast['alm*mit'] = all_yeast['alm']*all_yeast['mit']
all_yeast['vac/alm'] = all_yeast['vac']/all_yeast['alm']
all_yeast['alm*nuc'] = all_yeast['alm']*all_yeast['nuc']

yeast_data = preprocessing.scale(yeast_data, with_mean= True, with_std= False)


yeast_data = all_yeast[['mcg','gvh', 'alm', 'mit', 'vac', 'nuc', 'erl', 'pox', 'mcg/gvh', 'mcg*gvh', 'alm*vac', 'mit*nuc', 'alm/gvh',                        'mcg/alm','mcg*alm', 'mcg*mit', 'mcg*vac', 'mcg*nuc', 'alm*mit', 'vac/alm',                        'alm*nuc']].astype(float)


class_label = all_yeast['Class']







#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=mutual_info_classif, k=10)
fit = bestfeatures.fit(yeast_data,class_label)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(yeast_data.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(21,'Score'))  #print 10 best features





all_yeast = pd.read_csv('all_yeast.csv')
all_yeast.fillna(all_yeast.median()['mcg':'nuc'], inplace = True)


all_yeast['mcg/gvh'] = all_yeast['mcg']/all_yeast['gvh']
all_yeast['mcg*gvh'] = all_yeast['mcg']*all_yeast['gvh']
all_yeast['alm*vac'] = all_yeast['alm']*all_yeast['vac']
all_yeast['mit*nuc'] = all_yeast['mit']*all_yeast['nuc']
all_yeast['alm/gvh'] = all_yeast['alm']/all_yeast['gvh']
all_yeast['mcg/alm'] = all_yeast['mcg']/all_yeast['alm']
all_yeast['mcg*alm'] = all_yeast['mcg']*all_yeast['alm']
all_yeast['mcg*mit'] = all_yeast['mcg']*all_yeast['mit']
all_yeast['mcg*vac'] = all_yeast['mcg']*all_yeast['vac']
all_yeast['mcg*nuc'] = all_yeast['mcg']*all_yeast['nuc']
all_yeast['alm*mit'] = all_yeast['alm']*all_yeast['mit']
all_yeast['vac/alm'] = all_yeast['vac']/all_yeast['alm']
all_yeast['alm*nuc'] = all_yeast['alm']*all_yeast['nuc']

yeast_data = all_yeast[['alm', 'alm*vac', 'vac/alm', 'alm/gvh', 'mcg/alm', 'alm*nuc', 'gvh', 'alm*mit', 'mcg*nuc', 'mit*nuc',                       'mcg*gvh']].astype(float)

class_label = all_yeast['Class']
yeast_data = preprocessing.scale(yeast_data, with_mean= True, with_std= False)


X_train, X_test, y_train, y_test = train_test_split(yeast_data, class_label, train_size=2/3, test_size=1/3, random_state=42)


X_train = preprocessing.scale(X_train, with_mean= True, with_std= False)
X_test = preprocessing.scale(X_test, with_mean= True, with_std= False)




knn = neighbors.KNeighborsClassifier(n_neighbors = 7)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy is:", accuracy_score(y_test, y_pred))





from sklearn.cluster import KMeans



all_yeast = pd.read_csv('all_yeast.csv') 
all_yeast.fillna(all_yeast.median()['mcg':'nuc'], inplace = True)

yeast_data = all_yeast[['mcg','gvh', 'alm', 'mit', 'vac', 'nuc','erl', 'pox']].astype(float)
class_label = all_yeast['Class']

yeast_data = preprocessing.scale(yeast_data, with_mean= True, with_std= False)

X_train, X_test, y_train, y_test = train_test_split(yeast_data, class_label, train_size=2/3, test_size=1/3, random_state=42)

km = KMeans(n_clusters=4) #(k=3) k_means.fit(X_train)
km.fit(X_train)
l = km.labels_
X_train = pd.DataFrame(X_train) 
X_test = pd.DataFrame(X_test)

X_train['km'] = l
X_test['km'] = km.predict(X_test)

knn = neighbors.KNeighborsClassifier(n_neighbors = 7) 
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy is:", accuracy_score(y_test, y_pred))





from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC



all_yeast = pd.read_csv('all_yeast.csv') 
all_yeast.fillna(all_yeast.median()['mcg':'nuc'], inplace = True)

yeast_data = all_yeast[['mcg','gvh', 'alm', 'mit', 'vac', 'nuc','erl', 'pox']].astype(float)
class_label = all_yeast['Class']

yeast_data = preprocessing.scale(yeast_data, with_mean= True, with_std= False)

X_train, X_test, y_train, y_test = train_test_split(yeast_data, class_label, train_size=2/3, test_size=1/3, random_state=42)

km = KMeans(n_clusters=8) #(k=3) k_means.fit(X_train)
km.fit(X_train)
l = km.labels_
X_train = pd.DataFrame(X_train) 
X_test = pd.DataFrame(X_test)

X_train['km'] = l
X_test['km'] = km.predict(X_test)


sc = SVC()
sc.fit(X_train,y_train)
y_pred = sc.predict(X_test)

print("Accuracy is:", accuracy_score(y_test, y_pred))





# k-means clustering

from sklearn.cluster import KMeans

#Selecting the best features from previous experiments with k-NN
yeast_data = all_yeast[['mcg','gvh', 'alm', 'mit', 'vac', 'nuc','erl', 'pox']].astype(float)

class_label = all_yeast['Class']

#mean centering
yeast_data = preprocessing.scale(yeast_data, with_mean= True, with_std= False)

#applying k-means to all data
k_means = KMeans(n_clusters=3)  #(k=3)
k_means.fit(yeast_data)

#Create a new column and store the cluster labels
yeast_data['ClusterLabel'] = k_means.labels_







# Classification with k-NN using cluster labels from k-mean


yeast_data = yeast_data[['mcg','gvh', 'alm', 'mit', 'vac', 'nuc','ClusterLabel']].astype(float)

class_label = all_yeast['Class']

X_train, X_test, y_train, y_test = train_test_split(yeast_data, class_label, train_size=2/3, test_size=1/3, random_state=42)


X_train = preprocessing.scale(X_train, with_mean= True, with_std= False)
X_test = preprocessing.scale(X_test, with_mean= True, with_std= False)

# applying k-NN algorithm (keeping the best k form before: k=10)
knn = neighbors.KNeighborsClassifier(n_neighbors = 10)  
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy is:", accuracy_score(y_test, y_pred))





# k-means clustering


yeast_data = all_yeast[['mcg','gvh', 'alm', 'mit', 'vac', 'nuc','alm*vac','mit*nuc', 'mcg/gvh','ClusterLabel']].astype(float)


class_label = all_yeast['Class']
yeast_data = yeast_data[['mcg','gvh', 'alm', 'mit', 'vac', 'nuc','ClusterLabel']].astype(float)



k_means = KMeans(n_clusters=25)     #k = 25
k_means.fit(yeast_data)

yeast_data['ClusterLabel'] = k_means.labels_
    
    


class_label = all_yeast['Class']

X_train, X_test, y_train, y_test = train_test_split(yeast_data, class_label, train_size=2/3, test_size=1/3, random_state=42)


X_train = preprocessing.scale(X_train, with_mean= True, with_std= False)
X_test = preprocessing.scale(X_test, with_mean= True, with_std= False)

# applying k-NN algorithm (keeping the best k form before: k=10)
knn = neighbors.KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))






yeast_data = yeast_data[['mcg','gvh', 'alm', 'mit', 'vac', 'nuc','ClusterLabel']].astype(float)


class_label = all_yeast['Class']

X_train, X_test, y_train, y_test = train_test_split(yeast_data, class_label, train_size=2/3, test_size=1/3, random_state=42)


X_train = preprocessing.scale(X_train, with_mean= True, with_std= False)
X_test = preprocessing.scale(X_test, with_mean= True, with_std= False)


knn = neighbors.KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy is:", accuracy_score(y_test, y_pred))




"""
References:


[1]medium.com/@rtjeannier/combining-data-sets-with-fuzzy-matching-17efcb510ab2
[2]towardsdatascience.com/natural-language-processing-for-fuzzy-string-matching-with-python-6632b7824c49
[3]datatofish.com/compare-values-dataframes
[4]dataconomy.com/2015/04/implementing-the-five-most-popular-similarity-measures-in-python/
[5]python.gotrained.com/nltk-edit-distance-jaccard-distance/
[6]towardsdatascience.com/kmeans-clustering-for-classification-74b992405d0a
and workshop example solutions.

