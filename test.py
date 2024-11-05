from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import time
import agg
from colors import color
from datas import get_datas

pointcount, datas, target, data_2d, dataformat = get_datas()

fig = plt.figure()

##### Target clustering

plt.subplot(4, 3, 2)

plt.title('Target')
for i in range(10):
    plt.scatter(data_2d[target == i, 0], data_2d[target == i, 1], color=color[i],marker='o',edgecolor='k')

##### Our Agglomerative Clustering, Linkage = Average

plt.subplot(4, 3, 4)

# Run algorithm and time counting

time1 = time.process_time()
agg_hierarchical_clustering_a = agg.AgglomerativeHierarchicalClustering(dataformat, 10, 2)
agg_hierarchical_clustering_a.run_algorithm()
time2 = time.process_time()

clust_a = agg_hierarchical_clustering_a.print(target)

# Counting accuracy

labels_custom_a = [None] * len(dataformat)
for cluster_id, points in clust_a:
    for point in points:
        point_index = dataformat.index(point)
        labels_custom_a[point_index] = cluster_id
accuracy_agg_average = adjusted_rand_score(target, labels_custom_a)

# Plot

plt.title(f'Our Agglomerative Clustering Average Link Time: {str(time2-time1)}, ARI: {accuracy_agg_average:.2f}')
for id, points in clust_a:
    clustcolor = color[id % len(color)]
    for point in points:
        plt.scatter(point[0], point[1], color=clustcolor, marker='o', edgecolor='k')

##### Sklearn Agglomerative Clustering, Linkage = Average

plt.subplot(4, 3, 5)

# Run algorithm and time counting

time3 = time.process_time()
clusters = AgglomerativeClustering(n_clusters=10,linkage="average").fit_predict(datas[:int(pointcount)])
time4 = time.process_time()

# Counting accuracy and plot

plt.title(f'Sklearn Agglomerative Clustering Average Link Time: {str(time4-time3)}, ARI: {adjusted_rand_score(target, clusters):.2f}')
for i in range(10):
    plt.scatter(data_2d[clusters == i, 0], data_2d[clusters == i, 1],color=color[i], marker='o', edgecolor='k')

##### Sklearn Kmeans Clustering, Random state = 1

plt.subplot(4, 3, 6)

# Run algorithm and time counting

timeKM = time.process_time()
clusters = KMeans(n_clusters=10, random_state=1).fit_predict(datas[:int(pointcount)])
timeKM = time.process_time()

# Counting accuracy and plot

plt.title(f'Sklearn Kmeans Clustering Time: {str(timeKM-timeKM)}, ARI: {adjusted_rand_score(target, clusters):.2f}')
for i in range(10):
    plt.scatter(data_2d[clusters == i, 0], data_2d[clusters == i, 1],color=color[i], marker='o', edgecolor='k')

##### Our Agglomerative Clustering, Linkage = Single

plt.subplot(4, 3, 7)

# Run algorithm and time counting

time5 = time.process_time()
agg_hierarchical_clustering_s = agg.AgglomerativeHierarchicalClustering(dataformat, 10, 0)
agg_hierarchical_clustering_s.run_algorithm()
time6 = time.process_time()

clust_s = agg_hierarchical_clustering_s.print(target)

# Counting accuracy

labels_custom_s = [None] * len(dataformat)
for cluster_id, points in clust_s:
    for point in points:
        point_index = dataformat.index(point)
        labels_custom_s[point_index] = cluster_id
accuracy_agg_single = adjusted_rand_score(target, labels_custom_s)

# Plot

plt.title(f'Our Agglomerative Clustering Single Link Time: {str(time6-time5)}, ARI: {accuracy_agg_single:.2f}')
for id, points in clust_s:
    clustcolor = color[id % len(color)]
    for point in points:
        plt.scatter(point[0], point[1], color=clustcolor, marker='o', edgecolor='k')

##### Sklearn Agglomerative Clustering, Linkage = Single

plt.subplot(4, 3, 8)

# Run algorithm and time counting

time7 = time.process_time()
clusters = AgglomerativeClustering(n_clusters=10,linkage="single").fit_predict(datas[:int(pointcount)])
time8 = time.process_time()

# Counting accuracy and plot

plt.title(f'Sklearn Agglomerative Clustering Single Link Time: {str(time8-time7)}, ARI: {adjusted_rand_score(target, clusters):.2f}')
for i in range(10):
    plt.scatter(data_2d[clusters == i, 0], data_2d[clusters == i, 1],color=color[i], marker='o', edgecolor='k')

##### Sklearn Kmeans Clustering, Random state = 2

plt.subplot(4, 3, 9)

# Run algorithm and time counting

timeKM = time.process_time()
clusters = KMeans(n_clusters=10, random_state=2).fit_predict(datas[:int(pointcount)])
timeKM = time.process_time()

# Counting accuracy and plot

plt.title(f'Sklearn Kmeans Clustering Time: {str(timeKM-timeKM)}, ARI: {adjusted_rand_score(target, clusters):.2f}')
for i in range(10):
    plt.scatter(data_2d[clusters == i, 0], data_2d[clusters == i, 1],color=color[i], marker='o', edgecolor='k')

##### Our Agglomerative Clustering, Linkage = Complete

plt.subplot(4, 3, 10)

# Run algorithm and time counting

time9 = time.process_time()
agg_hierarchical_clustering_c = agg.AgglomerativeHierarchicalClustering(dataformat, 10, 1)
agg_hierarchical_clustering_c.run_algorithm()
time10 = time.process_time()

clust_c = agg_hierarchical_clustering_c.print(target)

# Counting accuracy

labels_custom_c = [None] * len(dataformat)
for cluster_id, points in clust_c:
    for point in points:
        point_index = dataformat.index(point)
        labels_custom_c[point_index] = cluster_id
accuracy_agg_complete = adjusted_rand_score(target, labels_custom_c)

# Plot

plt.title(f'Our Agglomerative Clustering Complete Link Time: {str(time10-time9)}, ARI: {accuracy_agg_complete:.2f}')
for id, points in clust_c:
    clustcolor = color[id % len(color)]
    for point in points:
        plt.scatter(point[0], point[1], color=clustcolor, marker='o', edgecolor='k')

##### Sklearn Agglomerative Clustering, Linkage = Complete

plt.subplot(4, 3, 11)

# Run algorithm and time counting

time11 = time.process_time()
clusters = AgglomerativeClustering(n_clusters=10,linkage="complete").fit_predict(datas[:int(pointcount)])
time12 = time.process_time()

# Counting accuracy and plot

plt.title(f'Sklearn Agglomerative Clustering Complete Link Time: {str(time12-time11)}, ARI: {adjusted_rand_score(target, clusters):.2f}')
for i in range(10):
    plt.scatter(data_2d[clusters == i, 0], data_2d[clusters == i, 1],color=color[i], marker='o', edgecolor='k')

##### Sklearn Kmeans Clustering, Random state = 3

plt.subplot(4, 3, 12)

# Run algorithm and time counting

timeKM = time.process_time()
clusters = KMeans(n_clusters=10, random_state=3).fit_predict(datas[:int(pointcount)])
timeKM = time.process_time()

# Counting accuracy and plot

plt.title(f'Sklearn Kmeans Clustering Time: {str(timeKM-timeKM)}, ARI: {adjusted_rand_score(target, clusters):.2f}')
for i in range(10):
    plt.scatter(data_2d[clusters == i, 0], data_2d[clusters == i, 1],color=color[i], marker='o', edgecolor='k')

##### Print results

plt.subplots_adjust(wspace=0.4, hspace=0.6)
plt.show()
