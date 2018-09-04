# Answers-generation-for-QA-system
A method delivers a collection of candidate answers in response to a query against a collection of unstructured data.
## System Overview
QA research attempts to deal with a wide range of question types including fact, list, definition and how. Access to answers is currently dominated by two paradigms: a knowledge graph that answers questions about what is in a collection of structurer records; and a cluster method that delivers a collection of candidate answers in response to a query against a collection of unstructured data. This method focuses on dealing with the second paradigm based on information detected from cluster algorithm.  

Given a customer’s question of which the keyword is not in Knowledge Graph, the direct answers from Knowledge Graph cannot provide adequate information for customers. Our objective is to provide explanatory notes for customers in order to improve entity accessibility. For example, if a customer asks ‘how many people does the refrigerator suitable for?’. On a semantic level, the question is about the volume of a refrigerator. But in fact, the machine cannot understand the potential meaning of the question. In this method, we can provide an auxiliary answer like ‘Total volume ranges from 22-30 cubic feet, and the refrigerator is suitable for a family of three’. 

In one embodiment, the method comprises receiving an input query; conducting a similar question identification model to detect similar questions in QA knowledge base. For each question, the clustering algorithm is performed on answers and a sub-answer set is selected from both answers data and comments data based on the valid value list. For each of the sub-answers, a candidate ranking model is performed based on sentence similarity measurement. Finally, the system returns K candidate answers with top K highest similarity.

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
