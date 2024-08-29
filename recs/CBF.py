import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def content_based(data, user_id='A00540411RKGTDNU543WS', num_recommend=5):
    # Tạo các dataframe cho tất cả sách và sách đã được đánh giá cao bởi active user
    Book_df = data.groupby(['Title'])[['Title', 'description']].first().reset_index(drop=True)
    User_df = data[(data['User_id'] == user_id) & (data['review/score'] == 5)].groupby(['Title'])[['User_id', 'Title', 'description']].first().reset_index(drop=True)

    # print(f'Số lượng sách user id {user_id} đã mua:', User_df.shape[0])

    # Tạo TF-IDF matrix cho tất cả sách
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(Book_df['description'].values)

    # Sử dụng NearestNeighbors để tìm sách tương đồng
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(tfidf_matrix)

    # Dictionary để lưu đánh giá dự đoán
    predicted_ratings = {}

    # Lặp qua các cuốn sách mà người dùng đã đánh giá cao
    for _, row in User_df.iterrows():
        title = row['Title']
        book_index = Book_df[Book_df['Title'] == title].index[0]  # Tìm index của sách trong Book_df
        # Tìm các cuốn sách tương tự
        distances, indices = model_knn.kneighbors(tfidf_matrix[book_index], n_neighbors=5)
        # Tính toán dự đoán đánh giá
        for dist, idx in zip(distances.flatten()[1:], indices.flatten()[1:]):
            similar_title = Book_df.iloc[idx]['Title']
            if similar_title not in User_df['Title'].values:  # Chỉ tính toán cho sách chưa đọc
                # print(dist)
                rating = 5 *(1 - dist)  # Công thức dự đoán
                if similar_title in predicted_ratings:
                    predicted_ratings[similar_title].append(rating)
                else:
                    predicted_ratings[similar_title] = [rating]

    # Tính trung bình đánh giá cho các sách nếu có nhiều đề xuất giống nhau
    final_predictions = {title: np.mean(ratings) for title, ratings in predicted_ratings.items()}

    # Sắp xếp các sách theo điểm dự đoán từ cao đến thấp
    sorted_predictions = sorted(final_predictions.items(), key=lambda x: -x[1])

    # Lấy ra N cuốn sách có điểm cao nhất
    recommendations = sorted_predictions[:num_recommend]

    # print('Các cuốn sách được đề xuất:')
    # for book, score in recommendations:
    #     print(f"Title: {book}, Predicted Score: {score}")

    return recommendations