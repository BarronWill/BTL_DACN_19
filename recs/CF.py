import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def colab_based(data, user_id ='A00540411RKGTDNU543WS', num_recommend=5):
  k_neighbors = 20
  # Tạo ma trận đánh giá
  book_features_df = data.pivot_table(index='User_id', columns='Title', values='review/score').fillna(0)[:100]
  book_features_df_matrix = csr_matrix(book_features_df.values)
  # print(book_features_df_matrix.shape)

  # Tìm ra k người dùng có mực độ tương đồng gần nhất với active user
  model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
  model_knn.fit(book_features_df_matrix)

  query_index = book_features_df.index.get_loc(user_id)
  # print('Query index: ', query_index)
  # print('User ID: ', user_id)

  distances, indices = model_knn.kneighbors(book_features_df.iloc[query_index].values.reshape(1, -1), n_neighbors=k_neighbors)
  neighbors_indices = indices[:k_neighbors]

  # Lấy ra các cuốn sách mà active_user chưa đọc (đánh giá = 0)
  active_user_ratings = book_features_df.iloc[query_index, :]
  unrated_items = active_user_ratings[active_user_ratings == 0].index

  predicted_ratings = {}
  # Dự đoán đánh giá cho mỗi cuốn sách chưa đọc
  for item_index in unrated_items:
      ratings_sum = 0
      count = 0
      for neighbor_index in indices[0]:
          neighbor_ratings = book_features_df.iloc[neighbor_index]
          if neighbor_ratings[item_index] != 0:  # Chỉ xem xét người dùng đã đánh giá
              ratings_sum += neighbor_ratings[item_index]
              count += 1

      if count == 0:
          predicted_ratings[item_index] = 0
      else:
          predicted_ratings[item_index] = ratings_sum / count

  # Recommend
  sorted_predictions = sorted(predicted_ratings.items(), key=lambda x: -x[1])
  recommendations = [(item_index, rating) for item_index, rating in sorted_predictions[:num_recommend]]

  return recommendations