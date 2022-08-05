from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from imports import *

"""permu_dataframe = pd.read_csv("rain_PFI.csv")

print(permu_dataframe.head(10))
important_variables = permu_dataframe[permu_dataframe.iloc[:,2]>0]
not_important_variables = permu_dataframe[permu_dataframe.iloc[:,2]==0]
neg_important_variables = permu_dataframe[permu_dataframe.iloc[:,2]<0]

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)
print(important_variables['Mean Importance'].shape)
X = important_variables['Mean Importance']
print("X",X.shape)
X = X.values.reshape(1, -1)
X = StandardScaler().fit_transform(X)

print("X", X)
db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_
print("db.labels",db.labels_)
print("db",db)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)



print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
print(
    "Adjusted Mutual Information: %0.3f"
    % metrics.adjusted_mutual_info_score(labels_true, labels)
)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))"""


testytest = pd.DataFrame([0,0.1,0.1,0.3])
print(testytest.diff())