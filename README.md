# Embedding-based Tabular Data Discovery using Query-By-Table

## Abstract

Query-By-Table (QBT) is a service that searches for table data to compare and identify based on a specific table entered as a user query, returning a list of table data sorted in descending order of relevance. This process can utilize embedding techniques to transform each table dataset into a single semantic vector, leveraging valuable information inherent in the vast table data, such as column names, column values, and metadata. This research proposes a technique that uses the table data itself, rather than natural language or keyword-level queries, to search for table datasets that can be merged in order of relevance to the query. Additionally, the method applies weights that consider the relative importance of columns within the query table compared to those in the target search table to enhance the performance of the search results. Users can perform join mergers with the retrieved table data to extract significant new information that was not present in the source table data.

---

Calculate the cosine similarity between each object in the embedded table data, and return a `ranked list` of the most similar tabular data to the input query tabular data.   

And make a weight, it has been made possible to consider `the relative importance` of the columns within the tabular data and reflect this in the calculation.

<p align="center">
  <img src="https://github.com/Godsihyeong/Table2Vec/assets/105179996/09bce2c5-d409-4a71-9298-57db303a6867" width="600">
</p>



# Query-By-Table 방식의 임베딩 기반 테이블 데이터 탐색 기법

## 초록

테이블에 의한 탐색 (Query-By-Table)은 사용자 질의로 입력되는 특정 테이블과 비교, 식별 대상이 되는 테이블 데이터를 검색하여 연관도에 따라 내림차순으로 정렬된 테이블 데이터 리스트를 반환하는 서비스이다. 이를 위해 방대한 테이블 데이터에 내재된 컬럼명, 컬럼 값, 메타데이터 등의 유용한 정보를 활용하여 각 테이블 데이터를 하나의 의미적 벡터로 변환하는 임베딩 (embedding) 기술을 활용할 수 있다. 본 연구는 사용자 질의로서 자연어 또는 키워드 수준이 아닌, 테이블 데이터 자체를 사용하여 그 질의와 연관된 순서대로 융합 가능한 테이블 데이터들을 탐색하는 기술을 제안한다. 또한, 질의 테이블과 비교되는 검색 대상 테이블 내부의 컬럼 간 상대적 중요도를 고려한 가중치를 적용하여 탐색 결과의 성능을 높이고자 한다. 사용자는 탐색된 테이블 데이터들에 대한 조인 융합을 수행하여 원천 테이블 데이터에서는 존재하지 않았던 유의미한 새로운 정보를 추출할 수 있다.

---

임베딩된 테이블 데이터 간의 코사인 유사도를 계산하고, 이 값을 사용하여 쿼리 테이블에 대한 연관도에 따라 정렬된 리스트를 반환.

또한, 이 과정에서 가중치를 적용하여 쿼리 테이블과 비교 대상 테이블 간의 컬럼 간 상대적 중요도를 반영하여 검색 결과의 성능을 향상.

<p align="center">
  <img src="https://github.com/Godsihyeong/Table2Vec/assets/105179996/09bce2c5-d409-4a71-9298-57db303a6867" width="600">
</p>


