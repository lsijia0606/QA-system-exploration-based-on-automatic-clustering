
# coding: utf-8


import sqlite3
import pandas as pd
import os
import sys
from pandas import DataFrame
import csv
import codecs
import numpy as np
import scipy.stats as stats
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import heapq 
from math import exp
from sklearn.neighbors import DistanceMetric
import re
import datetime
import requests
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer  

def is_valid_value(row):
    # Outlier
    re1 = r"^([!@#$%\^&*()_+{}\[\]:\";'<>?,./\-=《》{}（）0-9a-zA-Z]+)$" #only symbols 
    re3 = r"^(其它|其他|无|不支持|支持|有|没有|是|官网为准|未知|以官网信息为准|以官网参数为准|已官网为准|以官网为准|以官网数据为准|以官网信息为准|参见详情页|好|不好)$" #stopwords in chinese QA dataset
    re5 = r"^(\-|\+)?\d+(\.\d+)?$"
    #clean_value_regex = re.compile(r'('+re1+'|'+re3+'|'+re5+')');
    clean_value_regex = re.compile(r'('+re1+'|'+re5+')');

    if pd.isnull(row['value']):
        return True
    row['value'] = row['value'].strip() 
    res1 = re.search(clean_value_regex, row['value'])
    return (res1 != None) 

#Chinese words
def is_chinese_value(row):
    clean_value_regex_chinese = re.compile(u"[\u4e00-\u9fa5]+");
    if pd.isnull(row['value']):
        return True
    row['value'] = row['value'].strip() 
    res1 = re.search(clean_value_regex_chinese, row['value'])
    return (res1 != None) 

#Words
def is_word_value(row):
    clean_value_regex_word = re.compile(r'^[A-Za-z]+$');
    if pd.isnull(row['value']):
        return True
    row['value'] = row['value'].strip() 
    res1 = re.match(clean_value_regex_word, row['value'])
    return (res1 != None) 
    
def gainvector(filepath):
    """input: file path of word2vec result
    output: dataframe of word vector, array of vectors"""
        
    #read csv
    with codecs.open(filepath, encoding='utf-8') as f:
        PhoneValueVector = pd.read_table(f,names= ['value'])
        PhoneValueVector = PhoneValueVector['value'].str.split(',',n=1,expand = True)
        PhoneValueVector.columns = ['value','vector']
        
    nan_value = PhoneValueVector[PhoneValueVector['vector'].isnull().values==True]
    PhoneValueVector = PhoneValueVector[PhoneValueVector['vector'].isnull().values==False]
    PhoneValueVector.index = range(len((PhoneValueVector)))
        
    #Invalid words
    invalid = PhoneValueVector.apply(is_valid_value, axis=1)
    PhoneValueVector = PhoneValueVector[invalid==False]
    #Chinese
    chinese = PhoneValueVector.apply(is_chinese_value, axis=1)
    PhoneValueVector_Chinese_ = PhoneValueVector[chinese==True]
    PhoneValueVector_Chinese_.index = range(len((PhoneValueVector_Chinese_)))
    #Character
    word = PhoneValueVector_Chinese_.apply(is_word_value, axis=1)
    PVV_Word= PhoneValueVector_Chinese_[word==True]
    PVV_Word.index = range(len((PVV_Word)))
    for m in range(len(PhoneValueVector_Chinese_)):
        if len(PhoneValueVector_Chinese_['value'][m])==1:
            PhoneValueVector_Chinese_ = PhoneValueVector_Chinese_.drop([m])
            PhoneValueVector_Chinese = PhoneValueVector_Chinese_
        else:
            PhoneValueVector_Chinese = PhoneValueVector_Chinese_
    PhoneValueVector_Chinese.index = range(len(PhoneValueVector_Chinese))
    print('Number of Words with Vector',len(PhoneValueVector_Chinese_))
    print('Number of Word without Vector',len(nan_value))
    print('Valid Word Vectors',len(PhoneValueVector_Chinese))
    ValueVectorList_Chi = []
    for m in range(len(PhoneValueVector_Chinese)):
        ValueVectorList_Chi.append(eval(PhoneValueVector_Chinese['vector'][m]))
    ValueVectorArray_Chi = np.array(ValueVectorList_Chi)
    return PhoneValueVector_Chinese,ValueVectorArray_Chi

     
def pre_auto_cluster(PhoneValueVector_Chinese,ValueVectorArray_Chi,n_neighbor,plot):
    
    value_size,feasure_size = ValueVectorArray_Chi.shape
    nbrs = NearestNeighbors(n_neighbors=n_neighbor+1, algorithm='brute',metric = 'cosine').fit(ValueVectorArray_Chi)
    knn_matrix = nbrs.kneighbors(ValueVectorArray_Chi, return_distance=False)
    cosine_dist = (1-cosine_similarity(ValueVectorArray_Chi))
    #local density 
    k_dis_list = []
    for q in range(knn_matrix.shape[0]):
        k_dis = 0
        for p in range(1,knn_matrix.shape[1]):
            dis = cosine_dist[q][knn_matrix[q][p]]
            k_dis += dis
        k_dis_mean = (knn_matrix.shape[1]-1)/(k_dis +1)# sys.float_info.min
        k_dis_list.append(k_dis_mean)
    #density base distance
    min_dist_list = []
    dist_matrix = pairwise_distances(ValueVectorArray_Chi, Y=None, metric='cosine')
    k_dis_sort = sorted(enumerate(k_dis_list), key=lambda x:x[1])
    for n in range(len(k_dis_sort)):
        dist_higher_list = []
        for m in range(n+1,len(k_dis_sort)):
            dist_higher = dist_matrix[k_dis_sort[n][0]][k_dis_sort[m][0]]
            dist_higher_list.append({'value index':k_dis_sort[n][0],
                                         'shortest index':k_dis_sort[m][0],'dist':dist_higher})
        if len(dist_higher_list)>0:
            min_dist = min(dist_higher_list,key=lambda x: x['dist'])
            min_dist_list.append(min_dist)
        else:
            index = dist_matrix[k_dis_sort[n][0]].tolist().index(max(dist_matrix[k_dis_sort[n][0]]))
            
            max_dist = ({'value index':k_dis_sort[n][0],'shortest index':index,'dist':dist_matrix[k_dis_sort[n][0]][index]})
            min_dist_list.append( max_dist )
    ld_dbd = pd.DataFrame(min_dist_list)
    ld_dbd['local density'] = sorted(k_dis_list)
    ld_dbd = ld_dbd.sort_values(by = 'value index').reset_index(drop = True)
    
    value_name_list = []
    for j in ld_dbd['value index'].tolist():
        value_name_list.append(PhoneValueVector_Chinese.loc[j]['value'])
    ld_dbd['value name'] = value_name_list
    
    combine = np.array(k_dis_list)*np.array(ld_dbd['dist'].tolist())
    combine_list = ({'value index':list(range(combine.shape[0])),'combine':combine, 
                 'value name':PhoneValueVector_Chinese.loc[list(range(combine.shape[0]))]['value']})
        
    combine_df = pd.DataFrame(combine_list)
    combine_df.index = range(len(combine_df))
    combine_ = sorted(list(combine))
    combine_df = combine_df.sort_values(by = 'combine',ascending = False).reset_index(drop = True)
    if plot != 0:
        #plot to identify the cluster center and size
        plt.figure(figsize=(14,10)) 
        plt.scatter(list(range(len(combine_))),combine_)
        plt.show()
        diff_list =[]
        combine_sort = combine_df['combine'].tolist()
        for q in range(len(combine_df)-1):
            diff = abs(combine_sort[q+1]-combine_sort[q])
            diff_list.append(diff)  
        plt.figure(figsize=(14,8)) 
        plt.bar(list(range(plot)),diff_list[:plot])
        plt.show()
    return ld_dbd,combine_df 
    
def auto_cluster(filepath,number_of_cluster,n_neighbor,top_num,plot):
        
        PhoneValueVector_Chinese,ValueVectorArray_Chi  = gainvector(filepath)
        ld_dbd,combine_df = pre_auto_cluster(PhoneValueVector_Chinese,ValueVectorArray_Chi,
                                             n_neighbor,plot)
        #Start Clustering
        value_size,feasure_size = ValueVectorArray_Chi.shape
        nbrs_ = NearestNeighbors(n_neighbors=20, algorithm='brute',metric = 'cosine').fit(ValueVectorArray_Chi)
        knn_matrix_ = nbrs_.kneighbors(ValueVectorArray_Chi, return_distance=False)
        dist_matrix = pairwise_distances(ValueVectorArray_Chi, Y=None, metric='cosine')
        cluster = []
        centers_ = combine_df['value index'].tolist()[0:number_of_cluster]
        
        cluster_len1 = 0
        cluster_len2 = 1
        value_list = [] 
        while len(cluster)!=ValueVectorArray_Chi.shape[0]-number_of_cluster and cluster_len1 != cluster_len2:
            cluster_len1 = len(cluster)
            for j in range(len(PhoneValueVector_Chinese)):
                if PhoneValueVector_Chinese.loc[j]['value'] not in [k['value'] for k in cluster]:
                    if j not in centers_:
                        near_index = ld_dbd['shortest index'][j]
                        near_value = PhoneValueVector_Chinese.loc[near_index]['value']
                        if near_index in centers_:
                            cluster.append({'label': PhoneValueVector_Chinese.loc[near_index]['value'],
                                            'value': PhoneValueVector_Chinese.loc[j]['value'],
                                            'dist':dist_matrix[near_index][j]})
                            value_list.append(PhoneValueVector_Chinese.loc[near_index]['value'])
                            value_list.append(PhoneValueVector_Chinese.loc[j]['value'])
                        elif len([k for k in cluster if k['value']==near_value])>0:
                            cluster.append({'label': [k['label'] for k in cluster if k['value']==near_value][0],
                                            'value': PhoneValueVector_Chinese.loc[j]['value'],
                                            'dist':dist_matrix[near_index][j]})
                            value_list.append([k['label'] for k in cluster if k['value']==near_value][0])
                            value_list.append(PhoneValueVector_Chinese.loc[j]['value'])
                        else:
                            continue
            cluster_len2 = len(cluster)
        extra = [a for a in PhoneValueVector_Chinese['value'].tolist() if a not in list(set(value_list))]   
        print('END','# cluster values',len(list(set(value_list))),'# nan values',len(extra),'total',len(PhoneValueVector_Chinese))
        cluster_df = pd.DataFrame(cluster)
        cluster_df = cluster_df.sort_values(by=['label','dist'])
        result = []
        center_top_df = cluster_df.loc[cluster_df['label'].isin(combine_df['value name'][:number_of_cluster])]
        for i in center_top_df['label'].drop_duplicates():
            center_top = center_top_df.loc[center_top_df['label']==i]
            cluster_ = center_top['value'].tolist()[:top_num]
            result.append({'center':i,'cluster':cluster_,
                           'combine':combine_df.loc[combine_df['value name'] == i]['combine'].tolist()[0]})
        result.append({'center':0,'cluster':extra,
                      'combine':0})
        result_df = pd.DataFrame(result)
        result_df = result_df[['center','cluster','combine']]
        result_df = result_df.sort_values(by = 'combine',ascending = False).reset_index()
        return result_df,cluster_df
        
def print_dict(v, prefix=''):
        global key
        if isinstance(v, dict):
            for k, v2 in v.items():
                p2 = "{}['{}']".format(prefix, k)
                key = k
                print_dict(v2, p2)
        elif isinstance(v, list):
            for i, v2 in enumerate(v):
                p2 = "{}[{}]".format(prefix, i)
                print_dict(v2, p2)
        else:
            if key =='features':
                dic.append(str(v))
        return dic
    
def pos(filepath,number_of_cluster,n_neighbor,top_num,plot):
        cluster_top,cluster_all = auto_cluster(filepath,number_of_cluster,n_neighbor,top_num,plot)
        
        valid_center = []
        for i in cluster_top['center'].tolist():
            response = requests.get("http://10.153.152.253:5000/LexicalAnalyze?Type=json&Key=Lsdif238fj&action=headdown&Sentence=%s"%i)
            data = response.json()
            global dic 
            dic = []
            features = print_dict(data)
            res1 = [a for a in features if a in ['xC','MD','RC']]
            if len(res1) == 0:
                ret = [a for a in features if a in  ['N','nc','DE','NE','V','vc']]
                if len(ret) != 0:
                    valid_center.append(i)
        print('Valid Cluster Centers:',valid_center)
        print('Invalid Cluster Centers:',list(set(cluster_top['center'].tolist()).difference(set(valid_center))))
        return cluster_top,cluster_all, valid_center
        
def tfidf(filepath,filepath1,number_of_cluster,n_neighbor,top_num,plot,top_similar):
        cluster_top,cluster_all,valid_center = pos(filepath,number_of_cluster,n_neighbor,top_num,plot)
        qa_pair = pd.read_table(filepath1,sep='\t',names = ['question','parser'])
        qa_pair = qa_pair.sort_index().reset_index()#all QA tokens
        valid_ =[]
        for j in valid_center:
            cluster_list = cluster_top.loc[cluster_top['center']==j]['cluster'].item()
            cluster_list.append(j)
            valid_ = valid_+cluster_list
        qa_parser = qa_pair['parser'].str.split(',')
        qa_parser_renew =[]
        for i in range(len(qa_parser)):
            qa_renew = [a for a in qa_parser[i] if a in valid_]
            qa_parser_renew.append(','.join(qa_renew))
        #tfidf
        transformer = TfidfTransformer(smooth_idf=False)
        starttime = datetime.datetime.now()
        vectorizer = CountVectorizer() 
        X = vectorizer.fit_transform(qa_parser_renew)
        word = vectorizer.get_feature_names() #all words 
        counts =  X.toarray()#term frequency
        endtime1 = datetime.datetime.now()
        tfidf = transformer.fit_transform(X)
        tfidf_array = tfidf.toarray()
        endtime = datetime.datetime.now()
        print (endtime - starttime)
        
        np.set_printoptions(threshold=np.inf)
        value_size,feasure_size = tfidf_array.shape
        nbrs = NearestNeighbors(n_neighbors=top_similar+1, algorithm='brute',metric = 'cosine').fit(tfidf.toarray())
        knn_matrix = nbrs.kneighbors(tfidf_array, return_distance=False)
        cosine_dis = cosine_similarity(tfidf.toarray())

        sim_ques_all = []
        for x in range(len(qa_parser_renew)):
            sim_ques = []
            for q in range(knn_matrix.shape[1]-1):
                
                sim_ques.append((knn_matrix[x][q],qa_pair['question'][knn_matrix[x][q]],cosine_dis[x ][knn_matrix[x ][q]]))
            sim_ques_all.append(sim_ques)
        top_ = pd.DataFrame({'question':qa_pair['question'].tolist(),'Top similar':sim_ques_all})
        print(top_)
        return cluster_top,cluster_all,valid_center,top_
    
def main():
    path1 = '/data/vector_ALLQuestions.txt'#all value vectors
    path2 = '/data/allQAtokens.txt'#tokenized answers
    result_df,cluster_all = auto_cluster(path1,
                                           number_of_cluster,
                                           k_nearest_neighbor,
                                           top_num,
                                           plot)
    result_df,cluster_all,valid_center,top_ = tfidf(path1,
                                                      path2,
                                                      number_of_cluster,
                                                      n_neighbor,
                                                      top_n_of_each_cluster,
                                                      plot,
                                                      top_similar_sentence)

if __name__ == "__main__":
    main()

