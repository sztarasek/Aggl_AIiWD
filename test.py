from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
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
    plt.scatter(data_2d[target == i, 0], data_2d[target == i, 1], color=color[i],marker='o',edgecolor='k')

time1 = time.process_time()
agg_hierarchical_clustering_a = agg.AgglomerativeHierarchicalClustering(dataformat, 10, 2)
agg_hierarchical_clustering_a.run_algorithm()
time2 = time.process_time()

clust_a = agg_hierarchical_clustering_a.print(target)

plt.subplot(4, 2, 3)
labels_custom_a = [None] * len(dataformat)
for cluster_id, points in clust_a:
    for point in points:
        point_index = dataformat.index(point)
        labels_custom_a[point_index] = cluster_id
accuracy_agg_average = adjusted_rand_score(target, labels_custom_a)
plt.title(f'Our Agglomerative Clustering Average Link Time: {str(time2-time1)}, ARI: {accuracy_agg_average:.2f}')
for id, points in clust_a:
    clustcolor = color[id % len(color)]
    for point in points:
        plt.scatter(point[0], point[1], color=clustcolor, marker='o', edgecolor='k')
aa = target, [id for id, _ in clust_a]

plt.subplot(4, 2, 4)
time3 = time.process_time()
clusters = AgglomerativeClustering(n_clusters=10,linkage="average").fit_predict(datas[:int(pointcount)])
time4 = time.process_time()
plt.title(f'Sklearn Agglomerative Clustering Average Link Time: {str(time4-time3)}, ARI: {adjusted_rand_score(target, clusters):.2f}')
for i in range(10):
    plt.scatter(data_2d[clusters == i, 0], data_2d[clusters == i, 1],color=color[i], marker='o', edgecolor='k')


time5 = time.process_time()
agg_hierarchical_clustering_s = agg.AgglomerativeHierarchicalClustering(dataformat, 10, 0)
agg_hierarchical_clustering_s.run_algorithm()
time6 = time.process_time()

clust_s = agg_hierarchical_clustering_s.print(target)

plt.subplot(4, 2, 5)
labels_custom_s = [None] * len(dataformat)
for cluster_id, points in clust_s:
    for point in points:
        point_index = dataformat.index(point)
        labels_custom_s[point_index] = cluster_id
accuracy_agg_single = adjusted_rand_score(target, labels_custom_s)
plt.title(f'Our Agglomerative Clustering Single Link Time: {str(time6-time5)}, ARI: {accuracy_agg_single:.2f}')
for id, points in clust_s:
    clustcolor = color[id % len(color)]
    for point in points:
        plt.scatter(point[0], point[1], color=clustcolor, marker='o', edgecolor='k')


plt.subplot(4, 2, 6)
time7 = time.process_time()
clusters = AgglomerativeClustering(n_clusters=10,linkage="single").fit_predict(datas[:int(pointcount)])
time8 = time.process_time()
plt.title(f'Sklearn Agglomerative Clustering Single Link Time: {str(time8-time7)}, ARI: {adjusted_rand_score(target, clusters):.2f}')
for i in range(10):
    plt.scatter(data_2d[clusters == i, 0], data_2d[clusters == i, 1],color=color[i], marker='o', edgecolor='k')


time9 = time.process_time()
agg_hierarchical_clustering_c = agg.AgglomerativeHierarchicalClustering(dataformat, 10, 1)
agg_hierarchical_clustering_c.run_algorithm()
time10 = time.process_time()

clust_c = agg_hierarchical_clustering_c.print(target)

plt.subplot(4, 2, 7)
labels_custom_c = [None] * len(dataformat)
for cluster_id, points in clust_c:
    for point in points:
        point_index = dataformat.index(point)
        labels_custom_c[point_index] = cluster_id
accuracy_agg_complete = adjusted_rand_score(target, labels_custom_c)
plt.title(f'Our Agglomerative Clustering Complete Link Time: {str(time10-time9)}, ARI: {accuracy_agg_complete:.2f}')
for id, points in clust_c:
    clustcolor = color[id % len(color)]
    for point in points:
        plt.scatter(point[0], point[1], color=clustcolor, marker='o', edgecolor='k')


plt.subplot(4, 2, 8)
time11 = time.process_time()
clusters = AgglomerativeClustering(n_clusters=10,linkage="complete").fit_predict(datas[:int(pointcount)])
time12 = time.process_time()
plt.title(f'Sklearn Agglomerative Clustering Complete Link Time: {str(time12-time11)}, ARI: {adjusted_rand_score(target, clusters):.2f}')
for i in range(10):
    plt.scatter(data_2d[clusters == i, 0], data_2d[clusters == i, 1],color=color[i], marker='o', edgecolor='k')

plt.subplots_adjust(wspace=0.4, hspace=0.6)
plt.show()
