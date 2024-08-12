import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

data = pd.read_csv('Data/new_book_Rating_095.csv')
print(data.head())

# Check null
data.isnull().sum()

book_features_df = data.pivot_table(index='User_id', columns='Title', values='review/score').fillna(0)


book_features_df_matrix = csr_matrix(book_features_df.values)
print(book_features_df_matrix.shape)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(book_features_df_matrix)

query_index = np.random.choice(book_features_df.shape[0])
print(query_index)
distances, indices = model_knn.kneighbors(book_features_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)


active_user = book_features_df.iloc[query_index,:]
active_user.shape

# Recommend
similar_user_ratings = book_features_df.iloc[indices.flatten()[1]]
items_list = similar_user_ratings[(active_user > 0) & (similar_user_ratings >= 4)]
print("Sách được đề xuất là: ", items_list.index.tolist())