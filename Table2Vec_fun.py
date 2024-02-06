import numpy as np
import gensim
import re
import pandas as pd
import itertools

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

# cosine 유사도
# cosine 유사도
# cosine 유사도

def similarity(x, y):
    if np.linalg.norm(x) > 0 and np.linalg.norm(y):
        cos_similar = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        return cos_similar
    else:
        return 0

# 벡터화 함수
# 벡터화 함수
# 벡터화 함수

def vectorization(word):
    # 단어가 pretrained model에 있을 경우
    if str(word) in word2vec_model:
        return word2vec_model[str(word)]
    # 단어가 없을 경우 -> 결측치 처리 (후에 0으로 처리할 예정, 고유명사(사람 이름, 게임 이름), 의미없는 숫자, 단일 알파벳 등등..)
    else:
        return np.zeros(300)

# column head 전처리 함수
# column head 전처리 함수
# column head 전처리 함수

def remove(lst):
    cleaned_list = []
    for item in lst:
        # 괄호와 괄호 안의 내용 제거
        item_no_brackets = re.sub(r"\(.*?\)|\{.*?\}|\[.*?\]", "", item)
        # 특수문자 제거 (-:.,& 포함)
        item_cleaned = re.sub(r"[-:.,&]|[^a-zA-Z0-9\s]", "", item_no_brackets)
        cleaned_list.append(item_cleaned.strip())
    return cleaned_list

# Table2Vec
# Table2Vec
# Table2Vec

def table2vec(dataframe):

    row_number = dataframe.shape[0]
    col_number = dataframe.shape[1]

    df = dataframe.applymap(lambda x: re.sub(r'[-:.,&]', '', x) if isinstance(x, str) else x)
    df = df.applymap(lambda x: re.sub(r'\([^)]*\)', '', x) if isinstance(x, str) else x)

    df_2_list = df.values.tolist()

    df_list = list(itertools.chain.from_iterable(df_2_list))

    table_vector = []

    for entity in df_list:
        temp = 0
        try:
            for word in str(entity).split(' '):
                temp += vectorization(word)
            table_vector.append(temp)
        except:
            table_vector.append(vectorization(word))

    return [table_vector[col_number*n : col_number*(n+1)] for n in range(row_number)]

# 입력받은 table(또는 query로 들어갈수도)의 평균(중심) 계산하는 함수
# 입력받은 table(또는 query로 들어갈수도)의 평균(중심) 계산하는 함수
# 입력받은 table(또는 query로 들어갈수도)의 평균(중심) 계산하는 함수

def centroid(table):
    flatten_table = list(itertools.chain.from_iterable(table))
    length = len(flatten_table)
    return sum(flatten_table) / length

# list 단위로 벡터화 하는 함수
# list 단위로 벡터화 하는 함수
# list 단위로 벡터화 하는 함수

def list2vector(lst):
    lst = [str(entity) for entity in lst]

    cleaned_list = remove(lst)

    lst_vectors = []

    for entity in cleaned_list:
        vector_sum = 0
        try:
            for word in entity.split(' '):
                vector_sum += vectorization(word)
            lst_vectors.append(vector_sum)
        except:
            lst_vectors.append(vectorization(entity))

    return lst_vectors

#### Weights ###
#### Weights ###
#### Weights ###

class Weights:
    def __init__(self, query_df):
        self.query_df = query_df

    def h2h(self, table_df):
        # 각 dataframe의 columns을 list로 생성
        query_columns = list(self.query_df.columns)
        table_columns = list(table_df.columns)

        query_col_vector = list2vector(query_columns)
        table_col_vector = list2vector(table_columns)

        cos_list = []

        # query column vector의 i번째 요소가
        for i in range(len(table_col_vector)):
            similar_sum = 0
            # table column vector의 요소들과 얼마나 유사한지 유사도 계산해서 총합
            for j in range(len(query_col_vector)):
                similar_sum += similarity(table_col_vector[i], query_col_vector[j])
            cos_list.append(similar_sum)

        if sum(cos_list) != 0:
            return [round(weight/sum(cos_list), 4) for weight in cos_list]
        else:
            return [0 for _ in range(len(cos_list))]

    def h2t(self, table_df):
        # query column 호출
        query_columns = list(self.query_df.columns)
        # query columns 벡터화
        query_column_vectors = list2vector(query_columns)

        # table column 호출
        table_columns = list(table_df.columns)

        table_column_vectors = list2vector(table_columns)

        cos_list = []

        # table_df 전처리
        table_df = table_df.applymap(lambda x: re.sub(r'[-:.,&]', '', x) if isinstance(x, str) else x)
        table_df = table_df.applymap(lambda x: re.sub(r'\([^)]*\)', '', x) if isinstance(x, str) else x)

        for i in range(len(table_columns)):
            compare_column = list2vector(list(table_df[table_columns[i]]))
            compare_column = table_column_vectors[i] + compare_column
            compare_column_sum = sum(compare_column)

            cos_sum = 0

            for query in query_column_vectors:
                cos_sum += similarity(compare_column_sum, query)

            cos_list.append(cos_sum)

        if sum(cos_list) != 0:
            return [round(weight/sum(cos_list), 4) for weight in cos_list]
        else:
            return [0 for _ in range(len(cos_list))]

#### fusion ###
#### fusion ###
#### fusion ###

class Fusion:
    def __init__(self, query):
        self.query = query

    def early(self, table):
        centroid_query = centroid(self.query)
        centroid_table = centroid(table)
        return similarity(centroid_query, centroid_table)

    def late(self, table, category):
        flatten_query = list(itertools.chain.from_iterable(self.query))
        flatten_table = list(itertools.chain.from_iterable(table))

        cos_list = []

        for i in range(len(flatten_query)):

            for j in range(len(flatten_table)):

                result = similarity(flatten_query[i], flatten_table[j])

                cos_list.append(result)

        if category == 'avg':
            total_sum = sum(cos_list)
            n = len(cos_list)
            return total_sum / n

        elif category == 'max':
            maximum = max(cos_list)
            return maximum

        elif category == 'sum':
            return sum(cos_list)

        else:
            return 'category를 다시 지정해주세요.'

#### Weights using smoothing ###
#### Weights using smoothing ###
#### Weights using smoothing ###

class Weights_smoothing:
    def __init__(self, query_df):
        self.query_df = query_df

    def h2h(self, table_df):
        # 각 dataframe의 columns을 list로 생성
        query_columns = list(self.query_df.columns)
        table_columns = list(table_df.columns)

        query_col_vector = list2vector(query_columns)
        table_col_vector = list2vector(table_columns)

        cos_list = [1/len(query_columns) for _ in range(len(table_columns))]

        # query column vector의 i번째 요소가
        for i in range(len(table_col_vector)):
            similar_sum = 0
            # table column vector의 요소들과 얼마나 유사한지 유사도 계산해서 총합
            for j in range(len(query_col_vector)):
                similar_sum += similarity(table_col_vector[i], query_col_vector[j])
            cos_list[i] += similar_sum

        if sum(cos_list) != 0:
            return [round(weight/sum(cos_list), 4) for weight in cos_list]
        else:
            return [0 for _ in range(len(cos_list))]
        
    def h2t(self, table_df):
        # query column 호출
        query_columns = list(self.query_df.columns)
        # query columns 벡터화
        query_column_vectors = list2vector(query_columns)

        # table column 호출
        table_columns = list(table_df.columns)

        table_column_vectors = list2vector(table_columns)

        cos_list = [1/len(table_columns) for _ in range(len(table_columns))]

        # table_df 전처리
        table_df = table_df.applymap(lambda x: re.sub(r'[-:.,&]', '', x) if isinstance(x, str) else x)
        table_df = table_df.applymap(lambda x: re.sub(r'\([^)]*\)', '', x) if isinstance(x, str) else x)

        for i in range(len(table_columns)):
            compare_column = list2vector(list(table_df[table_columns[i]]))
            compare_column = table_column_vectors[i] + compare_column
            compare_column_sum = sum(compare_column)

            cos_sum = 0

            for query in query_column_vectors:
                cos_sum += similarity(compare_column_sum, query)

            cos_list[i] += cos_sum

        if sum(cos_list) != 0:
            return [round(weight/sum(cos_list), 4) for weight in cos_list]
        else:
            return [0 for _ in range(len(cos_list))]
