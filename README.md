# Query By Table

```
To retrieve tabular datas similar with a query tabular data, vectorize each data  
```

Table2Vec : Using Google's pre-trained word2vec model, change a entity in tabular data to vector.   

Calculate the cosine similarity between each object in the vectorized table data, and return a `ranked list` of the most similar tabular data to the input query tabular data.   

And make a weight, it has been made possible to consider `the relative importance` of the columns within the tabular data and reflect this in the calculation.

<p align="center">
  <img src="https://github.com/Godsihyeong/Table2Vec/assets/105179996/09bce2c5-d409-4a71-9298-57db303a6867" width="600">
</p>
