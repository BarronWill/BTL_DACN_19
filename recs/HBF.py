import numpy as np
import pandas as pd
from recs.CF import *
from recs.CBF import *

def weighted_hybrid(data,
                    colab_recommendations,
                    content_recommendations,
                    n_recommend = 5,
                    colab_weight=0.6,
                    content_weight=0.4):
  # colab_recommendations = colab_based(data, user_id)
  # content_recommendations = content_based(data, user_id)
  colab_recommendations = colab_recommendations
  content_recommendations = content_recommendations
  print('Colab Recommendations:')
  for rec, score in colab_recommendations:
    print(rec, '-', score)

  print('\nContent Recommendations:')
  for rec, score in content_recommendations:
      print(rec, '-', score)

  hybrid_recommendations = {}

  for book, score in colab_recommendations:
      hybrid_recommendations[book] = score * colab_weight

  for book, score in content_recommendations:
      if book in hybrid_recommendations:
          hybrid_recommendations[book] += score * content_weight
      else:
          hybrid_recommendations[book] = score * content_weight


  sorted_recommendations = sorted(hybrid_recommendations.items(), key=lambda x: x[1], reverse=True)

  return sorted_recommendations[:n_recommend]
