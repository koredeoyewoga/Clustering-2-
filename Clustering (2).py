#import packages and dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import seaborn as sb #package for plotting
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("country_data.csv") #read all the data from houseprice_data.csv
print(df.head()) #display the first 5 rows
print(df.isnull().sum()) #check if any cell is null
print(df.describe())

x = data = df[['child_mort', 'exports', 'health', 'imports', 'income',
        'inflation', 'life_expec','total_fer', 'gdpp']]
#y = df['price']
cor = x.corr()
plt.figure(figsize=(15, 13))
sb.heatmap(data = cor, annot = True, cmap='mako', center=0)
plt.savefig('Heatmap_diagram.png')
#here the

plt.figure(figsize=(10,6))
plt.scatter(df['income'], df['gdpp'])
plt.xlabel('income')
plt.ylabel('gdpp')
plt.title('Country Data')

plt.figure(figsize=(10,6))
plt.scatter(df['child_mort'], df['total_fer'])
plt.xlabel('total_fer')
plt.ylabel('child_mort')
plt.title('Country Data')

X = df[['child_mort', 'total_fer']].copy()

cluster_score = []
for i in range(1, 5):
    kmeans = KMeans(n_clusters=i, init='random', random_state=42)
    kmeans.fit(X)   
    cluster_score.append(kmeans.inertia_)
    
plt.figure(figsize=(10,6))
plt.plot(range(1, 5), cluster_score)
plt.xlabel('Number of Clusters')
plt.ylabel('Clustering Score')
plt.title('Elbow Method')
plt.show()

#fit the model and predict
kmeans = KMeans(n_clusters=2, random_state=42) # define the model
kmeans.fit(X)

# assign each data point to a cluster
prediction = kmeans.predict(X) 
print(prediction)

from numpy import unique
# get all of the unique clusters
dbscan_clusters = unique(prediction)
print(dbscan_clusters)

X['Clusters'] = pd.DataFrame(prediction, columns=['clusters'])
print(X.head())

sns.lmplot(x='child_mort', y='total_fer', data=X, fit_reg=False, hue='Clusters', legend=True)
plt.legend(loc='lower right')
plt.show()

