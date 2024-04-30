# Embedding-based Tabular Data Discovery using Query-By-Table

## Abstract

Query-By-Table (QBT) is a service that searches for table data to compare and identify based on a specific table entered as a user query, returning a list of table data sorted in descending order of relevance. This process can utilize embedding techniques to transform each table dataset into a single semantic vector, leveraging valuable information inherent in the vast table data, such as column names, column values, and metadata. This research proposes a technique that uses the table data itself, rather than natural language or keyword-level queries, to search for table datasets that can be merged in order of relevance to the query. Additionally, the method applies weights that consider the relative importance of columns within the query table compared to those in the target search table to enhance the performance of the search results. Users can perform join mergers with the retrieved table data to extract significant new information that was not present in the source table data.

---

Calculate the cosine similarity between each object in the vectorized table data, and return a `ranked list` of the most similar tabular data to the input query tabular data.   

And make a weight, it has been made possible to consider `the relative importance` of the columns within the tabular data and reflect this in the calculation.

<p align="center">
  <img src="https://github.com/Godsihyeong/Table2Vec/assets/105179996/09bce2c5-d409-4a71-9298-57db303a6867" width="600">
</p>
