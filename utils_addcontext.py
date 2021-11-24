import datasets

from datasets import load_dataset, load_metric

from typing import List
from typing import Dict
from typing import Tuple
from typing import Union

import pandas as pd
import numpy as np
import torch

def get_all_good_entries(qrels: Dict[str, Dict[str, int]], 
                         corpus: Dict[str, Dict[str, str]]) -> List[str]:
  '''
  Using qrels (query-answers dictionnary), go through all queries and for
  queries which are questions, analyze the contexts corresponding to the query.
  If the context is large enough and isn't added already, we add it in our list
  of context to add.
  Return a list of unique contexts from a dataset
  '''
  all_good_entries = []

  for i in qrels:
    # Check if query is a question, because we want a question-answer system
    # So it feels natural to get contexts related to questions
    if "QALD2" in i:
      # For every entry that correspond to a question
      for j in qrels[i]:
        # Get context
        context = corpus[j]['text']
        # Check if it's long enough for us
        context_tokens = context.split(' ')
        context_words = [word.lower() for word in context_tokens]
        if len(context_words) >= 50 and context not in all_good_entries:
          # Big enough and not yet added into our dataset
          # Add it in the list of good context
          all_good_entries.append(context)
  return all_good_entries

def transform_into_dico(d: datasets.dataset_dict.DatasetDict) -> Dict[str, Dict[str, str]]:
  '''
  Transform the SQuADv2 dataset into a dictionnary with unique contexts
  '''
  dico = {}
  for i in range(len(d['validation'])):
    context = d['validation'][i]['context']
    if context not in dico:
      new_dico = {}
      new_dico['title'] = d['validation'][i]['title']
      new_dico['question'] = d['validation'][i]['question']
      new_dico['answers'] = d['validation'][i]['answers']
      dico[context] = new_dico
  return dico

def create_list_of_all(dico: Dict[str, Dict[str, str]]) -> Union[List[str], List[str], List[str], List[str]]:
  '''
  From a dictionnnary formed with the function above, transform it into a list
  of questions, contexts, titles and answers
  '''
  all_questions = []
  all_contexts = []
  all_titles = []
  all_answers = []

  for context in dico.keys():
    all_contexts.append(context)
    all_questions.append(dico[context]['question'])
    all_titles.append(dico[context]['title'])
    all_answers.append(dico[context]['answers'])
  return all_questions, all_contexts, all_titles, all_answers

def get_all_unique_questions(all_questions: List[str],
                             df: pd.DataFrame) -> Union[List[str], Dict[str, List[int]]]:
  '''
  '''
  seen = set()
  seen_add = seen.add
  unique_questions = [x for x in all_questions if not (x in seen or seen_add(x))]
  q_a = {}

  for question in unique_questions:
    a = df.loc[df['question'] == question]
    l = list(a.index)
    q_a[question] = l

  return unique_questions, q_a