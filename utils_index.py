import pandas as pd
import numpy as np

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from datasets import load_dataset, load_metric

from typing import List
from typing import Dict
from typing import Tuple

from sentence_transformers import SentenceTransformer
import sentence_transformers.util

import faiss
from faiss import normalize_L2

### Functions for indexing with asymetric similarity models

def index_model_cosine_similarity(model, questions, contexts):
  '''
  Calculate the cosine similarity between given questions
  and given contexts by using a given model
  Returns a list of length len(questions) of list of len(contexts)
  of cosine similarity between question[i] and all contexts
  '''
  embeddings = model.encode(contexts, show_progress_bar=True)
  question_embeddings = model.encode(questions, show_progress_bar=True)
  similarity = sentence_transformers.util.pytorch_cos_sim(question_embeddings, embeddings)
  return similarity

def index_model_dot_product_similarity(model, questions, contexts):
  '''
  Calculate the dot product similarity between given questions
  and given contexts by using a given model
  Returns a list of length len(questions) of list of len(contexts)
  of dot product similarity between question[i] and all contexts
  '''
  embeddings = model.encode(contexts, show_progress_bar=True)
  question_embeddings = model.encode(questions, show_progress_bar=True)
  similarity = sentence_transformers.util.dot_score(question_embeddings, embeddings)
  return similarity

def get_ranks(res, unique_questions, q_a):
  '''
  From the list of list similarities, calculate the reciprocal rank for 
  each question
  Returns a list of reciprocal ranks corresponding to a question by their index
  in the list
  '''
  all_ranks = []

  for i in range(len(res)):
    similarities = res[i]
    index = sorted(range(len(similarities)), key=lambda k: similarities[k])
    index.reverse()
    question = unique_questions[i]
    for rank in range(len(index)):
      if index[rank] in q_a[question]:
        all_ranks.append(rank + 1)
        break
  return all_ranks

def get_top10_asymetric(similarities):
  '''
  Get the top 10 contexts who are the most similar to a 
  given question.
  List given as parameter is a list of cosine/dot product between all contexts
  and a given question
  '''
  index = sorted(range(len(similarities)), key=lambda k: similarities[k])
  index.reverse()
  return index[:10]

# To use with map()
def inverse(n):
  return 1/n

### Functions for indexing with nearest neighbors

def create_index(contexts, model, df):
  '''
  Use a sentence-transformer model to encode our contexts, and use
  faiss to index
  Returns that index
  '''
  embeddings = model.encode(contexts, show_progress_bar=True)

  embeddings = np.array([embedding for embedding in embeddings]).astype("float32")

  # Index with normalize_L2 to compute the similarities
  index = faiss.IndexFlatL2(embeddings.shape[1])
  index = faiss.IndexIDMap(index)
  index.add_with_ids(embeddings, df.index.values)

  return index

def doc_search(questions, model, index, num_results=10):
  '''
  For all questions given as parameters, search for the top k documents 
  corresponding to a given question
  Returns a list of distance and index corresponding to a certain context
  '''
  vector = model.encode(list(questions), show_progress_bar=True)
  D, I = index.search(np.array(vector).astype("float32"), k=num_results)
  return D, I

def MMR_test(res, unique_questions, q_a):
  '''
  Compute the MMR
  '''
  test = []

  for i in range(len(res)):
    for rank in range(len(res[i])):
      if res[i][rank] in q_a[unique_questions[i]]:
        test.append(rank + 1)
        break
  r = map(inverse, test)
  return sum(list(r)) / len(test)

def get_top10_context(question, model, index, contexts):
  list_context = []
  D, I = doc_search([question], model, index, num_results=10)
  for i in I[0]:
    list_context.append(contexts[i])
  return list_context

