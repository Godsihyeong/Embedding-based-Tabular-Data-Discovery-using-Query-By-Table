import re
import numpy as np
import gensim
from urllib.request import urlretrieve, urlopen

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

#########################
#########################
#########################

def cos_similar(x, y):
    if np.linalg.norm(x) > 0 and np.linalg.norm(y) > 0:
        cos = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        return cos
    else:
        return 0

def centroid(table):
    table_sum = 0
    
    len_table_row = len(table)
    len_table_col = len(table[0])
    
    for row in table:
        table_sum += np.sum(row, axis = 0)
    
    centroid_table = table_sum / (len_table_row * len_table_col)
    
    return centroid_table

def vectorization(word):
        if word in word2vec_model:
            return word2vec_model[word]
        else:
            return np.zeros(300)

#########################
#########################
#########################

# Preprocess and vectorization of tabular data

class Table2Vec:
    def __init__(self, df):
        self.df = df
        self.table_vector = []
        
    def table2vec(self):
        self.df = self.df.applymap(lambda x: re.sub(r'[-:.,&]', '', x) if isinstance(x, str) else x)
        self.df = self.df.applymap(lambda x: re.sub(r'\([^)]*\)', '', x) if isinstance(x, str) else x)

        df_2_list = self.df.values.tolist()

        for row in df_2_list:
            
            split = []

            for entity in row:
                temp = []
                
                try:
                    for word in entity.split(' '):
                        temp.append(str(word))
                    split.append(temp)
                except:
                    split.append([str(entity)])

            row_vector = [[vectorization(word) for word in entity] for entity in split]

            for i in range(len(row_vector)):
                row_vector[i] = np.sum(np.array(row_vector[i]), axis = 0)

            self.table_vector.append(row_vector)

        return self.table_vector

# Calculate query(also tabular) and table
          
class Retrieval:
    def __init__(self, query, table):
        self.query = query
        self.table = table
        
    def early_fusion(self):
        centroid_query = centroid(self.query)
        centroid_table = centroid(self.table)
        return cos_similar(centroid_query, centroid_table)
    
    def late_fusion(self, category):
        flatten_query = sum(self.query, [])
        flatten_table = sum(self.table, [])
        
        cos_list = []
        
        for i in range(len(flatten_query)):
            
            for j in range(len(flatten_table)):
                    
                cos_list.append(cos_similar(flatten_query[i], flatten_table[j]))
        
        if category == 'avg':
            total_sum = sum(cos_list)
            n = len(cos_list)
            return total_sum / n

        elif category == 'max':
            maximum = max(cos_list)
            return maximum

        else:
            return sum(cos_list)
                    