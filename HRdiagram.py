import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster as cls
import sklearn.metrics as met
import joblib 
df1 = pd.read_csv('Stars.csv')
df1['Temp_log10'] = np.log10(df1[['Temperature']])
X = df1['Temp_log10']
y = df1[['A_M']]
df2 = df1[['Temp_log10','A_M']]
kMeans = cls.KMeans(n_clusters=6,verbose=1,max_iter=300,random_state=42)
kMeans.fit(df2)
centroids = kMeans.cluster_centers_
labels = kMeans.labels_
centroid_X = centroids[:,0]
centroid_Y = centroids[:,1]
X_cluster = np.column_stack((X,y))
score = met.silhouette_score(X_cluster,labels,metric='euclidean')
print(f"Silhouette score: {score:.4f}")
joblib.dump(kMeans,'kMeans_model.sav')
plt.scatter(X,y,c=labels,cmap='rainbow')
plt.scatter(centroid_X,centroid_Y,color='black',marker='X',s=100)
plt.title('HR Diagram (X points are centroids, from Clustering algorithm)')
plt.xlabel('log10(Temperature), Temperature in K')
plt.ylabel('Absolute Luminosity (Mv)')
plt.savefig('HR_diagram.png')
plt.show()