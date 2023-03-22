# 层次聚类


#%%

import matplotlib.pyplot as plt  
import pandas as pd  
import numpy as np
import scipy.cluster.hierarchy as shc
customer_data = pd.read_csv(r'.\Shopping_CustomerData.csv')  


# 获取收入和支出
data = customer_data.iloc[:, 3:5].values  

#%%

plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(data, method='ward'))  
plt.savefig("./收支-01.jpg")
plt.show()
#%%

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cluster.fit_predict(data) 



plt.figure(figsize=(10, 7))  
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')  
plt.savefig("./收支-02.jpg")
plt.show()
