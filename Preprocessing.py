import numpy as np
import pandas as pd

# Read data
df_book = pd.read_csv('Data/books_data.csv', usecols=['Title', 'description'])
df_rating = pd.read_csv('Data/Books_rating.csv', usecols=['Title', 'User_id', 'review/score'])

# Drop all rows with na values
df_book = df_book.dropna()
df_rating = df_rating.dropna()

# Merge two table
merged_data = pd.merge(df_book, df_rating, on="Title", how="inner")

# Add Total Rating column
df_user_rating_count = (merged_data.groupby(['User_id'])
            .count()
            .sort_values('review/score', ascending=False)
            .reset_index()
            .rename(columns={'review/score':'TotalRating'}))[['User_id', 'TotalRating']]
df_merged = pd.merge(df_user_rating_count, merged_data, on='User_id', how='inner')
print(df_merged.shape)

# Remove the users's rating in 75% quantile
threshold = np.quantile(df_merged['TotalRating'], 0.75)
print(threshold)

df_new_merged = df_merged.query('TotalRating >= @threshold')
print(df_new_merged.shape)

df_new_merged.drop_duplicates(['User_id', 'Title'])
df_new_merged = df_new_merged.drop('TotalRating', axis=1)

# df_new_merged.to_csv('Data/bookRating.csv',index=False)

print(df_new_merged.head(5))
