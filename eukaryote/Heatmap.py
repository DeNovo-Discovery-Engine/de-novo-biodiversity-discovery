import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

clusters_csv = "/eukaryote/output/combined_clusters.csv"
cluster_df = pd.read_csv(clusters_csv)

cluster_counts = cluster_df['Cluster'].value_counts().sort_index()
print(cluster_counts)

heatmap_df = pd.DataFrame(cluster_counts).T 

plt.figure(figsize=(12, 2)) 
sns.heatmap(heatmap_df, annot=True, fmt="d", cmap="YlOrRd")
plt.title("Number of Sequences in Each Cluster")
plt.xlabel("Cluster ID")
plt.ylabel("Count")
plt.show()
