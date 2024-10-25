from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import time
import agg
import matplotlib.pyplot as plt

color = ["green", "blue", "black", "red", "grey", "pink", "purple", "lime", "aqua", "orange"]

pointcount = input("Amount of points to calculate type all for all points \n")

if pointcount == "all":
    pointcount = 1797

digits = load_digits()
datas = digits.data[:int(pointcount)]
target = digits.target[:int(pointcount)]

pca = PCA(n_components=2)
data_2d = pca.fit_transform(datas)
dataformat = data_2d.tolist()
print("Data converted to 2d. Time: "+str(time.process_time()))

fig = plt.figure()

plt.subplot(4, 2, 1)
plt.title('Target')
for i in range(10):
    plt.scatter(data_2d[target == i, 0], data_2d[target == i, 1], color=color[i],marker='o',edgecolor='k', cmap='viridis')

time1 = time.process_time()
agg_hierarchical_clustering = agg.AgglomerativeHierarchicalClustering(dataformat, 10, agg.average_link)
agg_hierarchical_clustering.run_algorithm()
time2 = time.process_time()

clust = agg_hierarchical_clustering.print(target)

plt.subplot(4, 2, 3)
plt.title('Our Agglomerative Clustering Average Link Time: ' + str(time2-time1))
for id, points in clust:
    clustcolor = color[id]
    for point in points:
        plt.scatter(point[0], point[1],color=clustcolor,marker='o',edgecolor='k', cmap='viridis')


plt.subplot(4, 2, 4)
time3 = time.process_time()
clusters = AgglomerativeClustering(n_clusters=10,linkage="average").fit_predict(datas[:int(pointcount)])
time4 = time.process_time()
plt.title('Sklearn Agglomerative Clustering Average Link Time: ' + str(time4-time3))
for i in range(10):
    plt.scatter(data_2d[clusters == i, 0], data_2d[clusters == i, 1],color=color[i], cmap='viridis', marker='o', edgecolor='k')


time5 = time.process_time()
agg_hierarchical_clustering = agg.AgglomerativeHierarchicalClustering(dataformat, 10, agg.single_link)
agg_hierarchical_clustering.run_algorithm()
time6 = time.process_time()

clust = agg_hierarchical_clustering.print(target)

plt.subplot(4, 2, 5)
plt.title('Our Agglomerative Clustering Single Link Time: ' + str(time6-time5))
for id, points in clust:
    clustcolor = color[id]
    for point in points:
        plt.scatter(point[0], point[1],color=clustcolor,marker='o',edgecolor='k', cmap='viridis')


plt.subplot(4, 2, 6)
time7 = time.process_time()
clusters = AgglomerativeClustering(n_clusters=10,linkage="single").fit_predict(datas[:int(pointcount)])
time8 = time.process_time()
plt.title('Sklearn Agglomerative Clustering Single Link Time: ' + str(time8-time7))
for i in range(10):
    plt.scatter(data_2d[clusters == i, 0], data_2d[clusters == i, 1],color=color[i], cmap='viridis', marker='o', edgecolor='k')


time9 = time.process_time()
agg_hierarchical_clustering = agg.AgglomerativeHierarchicalClustering(dataformat, 10, agg.complete_link)
agg_hierarchical_clustering.run_algorithm()
time10 = time.process_time()

clust = agg_hierarchical_clustering.print(target)

plt.subplot(4, 2, 7)
plt.title('Our Agglomerative Clustering Complete Link Time: ' + str(time10-time9))
for id, points in clust:
    clustcolor = color[id]
    for point in points:
        plt.scatter(point[0], point[1],color=clustcolor,marker='o',edgecolor='k', cmap='viridis')


plt.subplot(4, 2, 8)
time11 = time.process_time()
clusters = AgglomerativeClustering(n_clusters=10,linkage="complete").fit_predict(datas[:int(pointcount)])
time12 = time.process_time()
plt.title('Sklearn Agglomerative Clustering Complete Link Time: ' + str(time12-time11))
for i in range(10):
    plt.scatter(data_2d[clusters == i, 0], data_2d[clusters == i, 1],color=color[i], cmap='viridis', marker='o', edgecolor='k')

plt.show()
