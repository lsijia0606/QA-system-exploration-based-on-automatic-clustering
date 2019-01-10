# QA-system-exploration-based-on-automatic-clustering
This method proposes a new clustering algorithm that can detect the clustering centers and sizes automatically via density based metrics.

## System Overview
The clustering algorithm RLClu is proposed based on the assumption of “Cluster centers usually have a higher local density and a relative larger distance from objects with higher local densities”. It consists of three steps: metric extraction, clustering center identification, and object clustering.

•	Step 1. For each answer under a typical question, a parser is used to decompose it into keywords sequence. We only focus on the keywords that contain at least one Chinese word and its length should be bigger than 1. For example, “可以免费安装呀” will be decomposed to “可以//免费//安装//呀‘’, and ‘’呀’’ will be discarded;  
•	Step 2. Feed the keywords into the word embedding lookup to generate the vector for each word;  
•	Step 3. Calculate the similarity distance metric using SNN or cosine similarity and identify the nearest neighbor for each object;  
•	Step 4. Evaluate the centrality of objects based on K-density metric and minimum density-based distance;  
•	Step 5. Identify clustering centers and the number of clusters k by outward statistical testing. First, by sorting the product of the k-density and minimum density-based distance for each object in descending order, STClu generates a set of ordered statistics  . Then, find an obvious gap between this ordered statistics   from the largest end and choose top k nodes as cluster centers. Finally, the number of clustering centers is set as k and the objects corresponding to the first k objects are detected as the clustering centers;  
•	Step 6. Cluster the objects being not the clustering centers into the group containing its nearest neighbor with higher K-density.  

## Environment Setup
Please use Python 3.6 in the project. Install the dependencies via:  
  * spicy  
    &ensp;pip install scipy  
  * sklearn  
    &ensp;pip install -U scikit-learn  
     &ensp;or  
    &ensp;conda install scikit-learn  
## Experiment  
Run QAsystem.py  
* **Sample Input:**   
```python
path1 = '/SampleData/vector_Sample.txt' #all value vectors   
path2 = '/SampleData/SampleTokens.txt' #tokenized answers  
cluster_top,cluster_all = auto_cluster(path1, 50 ,10 ,20 ,0) # function to run cluster algorithm  
```
OR
```
cluster_top,cluster_all,valid_center,top_ = tfidf(path1,path2,50, 10, 20, 0,100) # function to run cluster algorithm, parser system and similarity matching  
```
**Important**: The Parser System here is an internal engine in our company and has not been officially online. We use Parser System to find the part of speech of each word. Therefore, you can use POS tagger instead when actually running the model.
* **Input Index:**  
&emsp;1.  number_of_cluster  
&emsp;2.  k_nearest_neighbor: number of nearest neighborhood when calculating the K-density.  
&emsp;3.  top_n_of_each_cluster: Choose the top n from each cluster results based on similarity distance from centers.  
&emsp;4.  plot: Whether to show the graph. Yes: input a number to represent the length of x_axle; No: 0  
&emsp;5.  top_similar_sentence: similar sentences for each question   
* **Result Explanation:**  
&emsp;1.  result_df: cluster results for top n values in each cluster  
&emsp;2.  cluster_all: cluster result for whole values in each cluster  
&emsp;3.  valid_center: valid center list after POS tag  
&emsp;4.  top_: top m similar sentence for each question  

