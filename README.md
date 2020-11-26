## ALGORITHM STEPS
1. Loading Dataset
   1. 20 News Group Dataset
   2. 20 Different Topics
2. Proccessing Dataset
   1. Removing Punctuation & numbers
   2. Extracting The Words
   3. Stop Word Removal
   4. Word Stemming
   5. Word Lemmatization
3. Training Topic Extractor from Processed Dataset (Or Loading Existing One)
5. Topic Retreival
6. Training Doc2Vec Model On Non Pre-Processed Data (Or Loading Existing One)
7. Application of Doc2Vec Model on Topics
8. Application of Doc2Vec On Each Document
9. Calculating the similarity between every pair dei(document vector) and tei(topic vector)

## TODO
1. Graph the vector representation of every document with its respective topics