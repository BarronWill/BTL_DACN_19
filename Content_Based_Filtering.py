import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel

data = pd.read_csv('Data/new_book_Rating_095.csv')
# print(data.head())

# Check null
data.isnull().sum()

Book_df = data.groupby(['Title'])[['Title', 'description']].first().reset_index(drop=True)
# print(Book_df.head())

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(Book_df['description'].values)
tfidf_matrix = tfidf_matrix.toarray()

sig  = sigmoid_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(Book_df.index, index=Book_df['Title']).drop_duplicates()
# Chọn ngẫu nhiên 1 cuốn sách
index = indices['\"A\" IS FOR ALIBI']
# Chọn top 10 cuốn sách gần giống nhất
recList = sorted(list(enumerate(sig[index])), key=lambda x: x[0], reverse=True)[1:10]

#Recommend
print("Goi y cho cuon sach ", Book_df['Title'].iloc[index])
for i in recList:
  print(Book_df['Title'].iloc[i[0]])