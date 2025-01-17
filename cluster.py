
# %%
#Import all the necessary packages

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pylab as plt
import seaborn as sns

#to scale the data using z-score 
from sklearn.preprocessing import StandardScaler

#importing clustering algorithms
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

#Silhouette score
from sklearn.metrics import silhouette_score


from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
     page_title='Energy Clustering App',
     page_icon='🌍',
     layout='wide',
     initial_sidebar_state='expanded')

# Title of the app
st.title('🌍 Energy Clustering App')

st.header ("Group Members") 
st.write("MICHAEL ASIEDU ACHEAMPONG ---11410558")
st.write("KWADWO JECTEY NYARKO ---11410422(Group Leader)")
st.write("BARIMA OWIREDU ADDO --- 22254055")
st.write("WILLIAM KWAME SAKA --- 22253219")


energy_17v = pd.read_excel("energy_for_45_European_countries.xlsx")


# Scale the features using MinMaxScaler
#scaler = MinMaxScaler()
#energy_17v_scaled = scaler.fit_transform(energy_17v.iloc[:,1:16])

st.markdown("-----------------")

st.sidebar.markdown("### 🧑‍💻 Clustering Method Selection")

st.sidebar.subheader('Choose clustering method')
clustering_method = st.sidebar.selectbox(('Method'),
                     ("kmeans","ward","single","complete","average","centroid"),
                     help="Select the algorithm you wish to use for clustering. Each method has different use cases.")

st.sidebar.markdown("### 🔢 Number of Clusters")

st.sidebar.subheader("Choose Number of Clusters")
number_clusters = st.sidebar.slider('Cluster Level',2,7,5,
                                    help="Adjust the number of clusters for the chosen method.")
##st.subheader('Correlation Heat Map')
##plt.figure(figsize  = (15,15))
###sns.heatmap(energy_17v.corr(numeric_only=True), annot = True, cmap="YlGnBu")
###st.pyplot(plt)

if clustering_method == "kmeans":
        scaler = StandardScaler()
        energy_17v_scaled = scaler.fit_transform(energy_17v.iloc[:, 1:16])
    
        st.header("K-Means Clustering")
        st.write("K-means clustering is an unsupervised machine learning algorithm used to group data points into a predefined number of clusters based on their similarity. It works by initializing cluster centroids, assigning each data point to the nearest centroid, and recalculating the centroids iteratively to minimize the within-cluster variance.")
    
        st.markdown("---")
    
    # KMeans with chosen number of clusters
        kmeans = KMeans(n_clusters=number_clusters, random_state=1)
        energy_17v['cluster'] = kmeans.fit_predict(energy_17v_scaled)

        st.subheader("Cluster Assignment for Countries:")
        st.dataframe(energy_17v[['Country', 'cluster']].sort_values(by='cluster', ascending=True), use_container_width=True)

        st.markdown("---")

    # Number of Countries in Each Cluster
        st.subheader("Number of Countries in Each Cluster:")
        cluster_counts = energy_17v.groupby('cluster')['Country'].count()

    # Cluster Counts Bar Chart
        fig, ax = plt.subplots(figsize=(8, 6))
        cluster_counts.plot(kind='bar', ax=ax, edgecolor='black')

    # Add data labels on bars
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', 
                    color='black', fontsize=12)

    # Customizing the plot
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', which='both', length=0)    

        plt.title('Number of Countries per Cluster', fontsize=16)
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Number of Countries', fontsize=12)
        st.pyplot(fig)
        st.markdown("---")

    # SSE (Sum of Squared Errors) for different K values
        sse = {}
        for k in range(2, 8):
            kmeans = KMeans(n_clusters=k, random_state=1).fit(energy_17v_scaled)
            sse[k] = kmeans.inertia_
        st.subheader("Elbow Method for Optimal K")
        plt.figure(figsize=(8, 6))
        plt.plot(list(sse.keys()), list(sse.values()), 'bx-', markersize=8)
        plt.title("Elbow Method for Optimal K", fontsize=16)
        plt.xlabel("Number of Clusters", fontsize=12)
        plt.ylabel("SSE (Sum of Squared Errors)", fontsize=12)
        plt.grid(False)
        st.pyplot(plt)
        st.markdown("---")

    # Silhouette Score for different K values
        sc = {}
        for k in range(2, 7):
            kmeans = KMeans(n_clusters=k, random_state=1).fit(energy_17v_scaled)
            labels = kmeans.predict(energy_17v_scaled)
            sc[k] = silhouette_score(energy_17v_scaled, labels)

        st.subheader("Silhouette Score for Optimal K")
        plt.figure(figsize=(8, 6))
        plt.plot(list(sc.keys()), list(sc.values()), 'bx-', markersize=8)
        plt.title("Silhouette Score for Optimal K", fontsize=16)
        plt.xlabel("Number of Clusters", fontsize=12)
        plt.ylabel("Silhouette Score", fontsize=12)
        plt.grid(False)
        st.pyplot(plt)

    
elif clustering_method == "ward":
    st.header("Ward's Method - Hierarchical Clustering")
    st.write("Ward's method is a hierarchical clustering technique that minimizes the variance within clusters as they are merged. Starting with each data point as its own cluster, the algorithm iteratively combines clusters in a way that results in the smallest increase in total within-cluster variance. This approach produces a dendrogram, which visually represents the process of clustering and helps in deciding the optimal number of clusters. Ward's method is particularly effective when clusters are roughly spherical and of similar size, making it widely used in fields like social sciences, bioinformatics, and market research. Its primary strength lies in creating well-defined and compact clusters.")
    st.markdown("-----------------")

    ##Heat Map
    st.subheader('Heat Map')
    plt.figure(figsize  = (15,15))
    sns.heatmap(energy_17v.corr(numeric_only=True), annot = True, cmap="YlGnBu")
    st.pyplot(plt)

    ##Scale the data using MinMax
    scaler = MinMaxScaler()
    energy_17v_scaled = scaler.fit_transform(energy_17v.iloc[:,1:16])

    # Ward's Method Dendrogram
    linked = linkage(energy_17v_scaled, method='ward')

    st.markdown("-----------------")
    # Plot dendrogram
    st.subheader("Ward's Method Dendogram")
    plt.figure(figsize=(10, 6), dpi=200)
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    st.pyplot(plt)

    # Calculate linkage matrix
    cluster_labels = fcluster(linked,number_clusters, criterion='maxclust')
    energy_17v['cluster'] = cluster_labels

    st.markdown("-----------------")
    # Show clusters
    st.subheader("Cluster Assignment for Countries:")
    st.dataframe(energy_17v[['Country', 'cluster']].sort_values(by='cluster', ascending=True),use_container_width=True)
    
    st.markdown("-----------------")

    st.subheader("Number of Countries in Each Cluster:")
    cluster_counts = energy_17v.groupby('cluster')['Country'].count()
    fig, ax = plt.subplots()
    cluster_counts.plot(kind='bar', ax=ax, edgecolor='black')
    
    # Add data labels on the bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='bottom',  # Adjusted va to 'bottom' for positioning above the bars
                color='black', fontsize=12)
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)    
    
    plt.title('Number of Countries per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Countries')
    st.pyplot(fig)

   
elif clustering_method == "single":
    st.header("Single Linkage - Hierarchical Clustering")
    st.write("Single linkage is a hierarchical clustering method that merges clusters based on the minimum distance between any two points from different clusters. This approach prioritizes the closest pair of points when forming clusters, leading to elongated or chain-like clusters in some cases. It is particularly useful for detecting clusters of arbitrary shapes but can be sensitive to noise and outliers. The method generates a dendrogram, enabling a visual representation of the clustering process and allowing users to determine the appropriate number of clusters. Single linkage is commonly applied in spatial data analysis and pattern recognition tasks.")
    st.markdown("-----------------")

    ##Heat Map
    st.subheader('Heat Map')
    plt.figure(figsize  = (15,15))
    sns.heatmap(energy_17v.corr(numeric_only=True), annot = True, cmap="YlGnBu")
    st.pyplot(plt)
 
    ##Scale the data using MinMax
    scaler = MinMaxScaler()
    energy_17v_scaled = scaler.fit_transform(energy_17v.iloc[:,1:16])


    st.markdown("-----------------")
    # Single Linkage Dendrogram
    linked = linkage(energy_17v_scaled, method='single')
   
    # Plot dendrogram
    st.subheader("Single Linkage Method Dendogram")
    plt.figure(figsize=(10, 6), dpi=200)
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    st.pyplot(plt)

    st.markdown("-----------------")
    # Calculate linkage matrix
    cluster_labels = fcluster(linked,number_clusters, criterion='maxclust')
    energy_17v['cluster'] = cluster_labels
    
    # Show clusters
    st.header("Cluster Assignment for Countries:")
    st.dataframe(energy_17v[['Country', 'cluster']].sort_values(by='cluster', ascending=True),use_container_width=True)

    st.markdown("-----------------")
    st.header("Number of Countries in Each Cluster:")
    cluster_counts = energy_17v.groupby('cluster')['Country'].count()
    fig, ax = plt.subplots()
    cluster_counts.plot(kind='bar', ax=ax, edgecolor='black')

    # Add data labels on the bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='bottom',  # Adjusted va to 'bottom' for positioning above the bars
                color='black', fontsize=12)
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)    
    
    plt.title('Number of Countries per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Countries')
    st.pyplot(fig)


elif clustering_method == "complete":
    st.header("Complete Linkage - Hierarchical Clustering")
    st.write("Complete linkage is a hierarchical clustering method that merges clusters based on the maximum distance between any two points from different clusters. This approach ensures that all points within a cluster are relatively close to each other, resulting in compact and spherical clusters. It is less sensitive to noise and outliers compared to single linkage, but it may overestimate distances between clusters. The method produces a dendrogram, which visually represents the clustering process and helps identify the optimal number of clusters. Complete linkage is commonly used in applications where uniform cluster shapes are desired.")
    st.markdown("-----------------")

    ##Heat Map
    st.subheader('Heat Map')
    plt.figure(figsize  = (15,15))
    sns.heatmap(energy_17v.corr(numeric_only=True), annot = True, cmap="YlGnBu")
    st.pyplot(plt)
    st.markdown("-----------------")
   
    ##Scale the data using MinMax
    scaler = MinMaxScaler()
    energy_17v_scaled = scaler.fit_transform(energy_17v.iloc[:,1:16])

    # Complete Linkage Dendrogram
    linked = linkage(energy_17v_scaled, method='complete')
    
    # Plot dendrogram
    st.subheader("Complete Linkage Method Dendogram")
    plt.figure(figsize=(10, 6), dpi=200)
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    st.pyplot(plt)
    st.markdown("-----------------")

    # Calculate linkage matrix
    cluster_labels = fcluster(linked,number_clusters, criterion='maxclust')
    energy_17v['cluster'] = cluster_labels
    
    # Show clusters
    st.write("Cluster Assignment for Countries:")
    st.dataframe(energy_17v[['Country', 'cluster']].sort_values(by='cluster', ascending=True),use_container_width=True)
    st.markdown("-----------------")

    # Number of countries per cluster
    st.subheader("Number of Countries in Each Cluster:")
    cluster_counts = energy_17v.groupby('cluster')['Country'].count()
    fig, ax = plt.subplots()
    cluster_counts.plot(kind='bar', ax=ax, edgecolor='black')
    
    # Add data labels on the bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='bottom',  # Adjusted va to 'bottom' for positioning above the bars
                color='black', fontsize=12)
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)    
    
    plt.title('Number of Countries per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Countries')
    st.pyplot(fig)

elif clustering_method == "average":
    st.write("Average Linkage - Hierarchical Clustering")
    st.write("Average linkage is a hierarchical clustering technique that merges clusters based on the average distance between all pairs of points from two different clusters. This method strikes a balance between single and complete linkage, producing clusters that are neither overly elongated nor overly compact. It works well in scenarios where clusters of varying shapes and sizes need to be identified. The process generates a dendrogram, which provides a visual representation of the clustering hierarchy and helps determine the optimal number of clusters. Average linkage is particularly useful for datasets with moderate levels of noise and outliers.")
    st.markdown("-----------------")
    
    ##Heat Map
    st.subheader('Heat Map')
    plt.figure(figsize  = (15,15))
    sns.heatmap(energy_17v.corr(numeric_only=True), annot = True, cmap="YlGnBu")
    st.pyplot(plt)
    st.markdown("-----------------")

    ##Scale the data using MinMax
    scaler = MinMaxScaler()
    energy_17v_scaled = scaler.fit_transform(energy_17v.iloc[:,1:16])

    # Average Linkage Dendrogram
    linked = linkage(energy_17v_scaled, method='average')
    
    # Plot dendrogram
    st.subheader("Average Linkage Method Dendogram")
    plt.figure(figsize=(10, 6), dpi=200)
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    st.pyplot(plt)

    # Calculate linkage matrix
    cluster_labels = fcluster(linked,number_clusters,criterion='maxclust')
    energy_17v['cluster'] = cluster_labels
    
    # Show clusters
    st.subheader("Cluster Assignment for Countries:")
    st.dataframe(energy_17v[['Country', 'cluster']].sort_values(by='cluster', ascending=True),use_container_width=True)
    st.markdown("-----------------")

    # Number of Countries in Each Cluster
    st.subheader("Number of Countries in Each Cluster:")
    cluster_counts = energy_17v.groupby('cluster')['Country'].count()
    fig, ax = plt.subplots()
    cluster_counts.plot(kind='bar', ax=ax, edgecolor='black')
    
    # Add data labels on the bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='bottom',  # Adjusted va to 'bottom' for positioning above the bars
                color='black', fontsize=12)
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)    
    
    plt.title('Number of Countries per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Countries')
    st.pyplot(fig)

elif clustering_method == "centroid":
    st.header("Centroid Linkage - Hierarchical Clustering")
    st.write("Centroid linkage is a hierarchical clustering method that calculates the distance between clusters based on the distance between their centroids, or geometric centers. Each time two clusters are merged, the new cluster's centroid is recalculated, and subsequent distances are based on this updated center. This method is effective for creating well-separated clusters but can sometimes produce results that are not hierarchical, as clusters may shift or split during the process. Centroid linkage is often used when the overall shape and balance of clusters are important, and it provides a dendrogram for visual analysis of the clustering process.")
    st.markdown("-----------------")
    
    ##Heat Map
    st.subheader('Heat Map')
    plt.figure(figsize  = (15,15))
    sns.heatmap(energy_17v.corr(numeric_only=True), annot = True, cmap="YlGnBu")
    st.pyplot(plt)
    st.markdown("-----------------")

    ##Scale the data using MinMax
    scaler = MinMaxScaler()
    energy_17v_scaled = scaler.fit_transform(energy_17v.iloc[:,1:16])

    # Centroid Linkage Dendrogram
    linked = linkage(energy_17v_scaled, method='centroid')
    
    # Plot dendrogram
    st.subheader("Centroid Linkage Method Dendogram")
    plt.figure(figsize=(10, 6), dpi=200)
    dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    st.pyplot(plt)
    st.markdown("-----------------")

    # Calculate linkage matrix
    cluster_labels = fcluster(linked,number_clusters, criterion='maxclust')
    energy_17v['cluster'] = cluster_labels
    
    # Show clusters
    st.subheader("Cluster Assignment for Countries:")
    st.dataframe(energy_17v[['Country', 'cluster']].sort_values(by='cluster', ascending=True),use_container_width=True)
    st.markdown("-----------------")
    
    # Number of countries per cluster
    st.subheader("Number of Countries in Each Cluster:")
    cluster_counts = energy_17v.groupby('cluster')['Country'].count()
    fig, ax = plt.subplots()
    cluster_counts.plot(kind='bar', ax=ax, edgecolor='black')
    
    # Add data labels on the bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height}', 
                (p.get_x() + p.get_width() / 2., height), 
                ha='center', va='bottom',  # Adjusted va to 'bottom' for positioning above the bars
                color='black', fontsize=12)
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)    
    
    plt.title('Number of Countries per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Countries')
    st.pyplot(fig)
