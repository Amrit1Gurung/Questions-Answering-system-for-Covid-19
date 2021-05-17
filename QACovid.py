import pandas as pd
import numpy as np
import os 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import pickle
import torch
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering
from rank_bm25 import BM25Okapi # Search engine

#download them.Necessary for stopwords removal and word or sentences tokenization
nltk.download('stopwords')
nltk.download('punkt')


# if gpu is available use that else use cpu
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = model.to(torch_device)


#load data
data = pd.read_csv('data/metadata.csv', low_memory=False)
data = data.dropna(subset=['abstract','title'])
data.drop_duplicates(subset=['title','abstract'],inplace=True)
data = data.reset_index()

stop_words = set(stopwords.words('english'))
# function to preprocess and remove stopwords
def preprocess(text):
  no_stops = []
  sent = text.lower()
  sent = re.sub(r"[^a-zA-Z]"," ",sent)    # removes all characters except alphabets
  sent = re.sub(r'\s+', ' ', sent)       # removes extra white spaces
  #sent = str(text)
  for w in sent.split():
    if not w in stop_words:
      if len(w)>1 and not w.isnumeric():
        no_stops.append(w)
  return no_stops

# nothing special here. we just take two columns we need; title and abstract than we join them and do preprocessing and find bm25 
def init(data):
  df = data.copy()
  df = data[['title','abstract']]
  df['title_abstract'] = data.abstract + " " + df.title
  df['title_abstract'] = df['title_abstract'].apply(lambda x: preprocess(x))
  bm25 = BM25Okapi(df.title_abstract.tolist())
  return bm25, df

# for a given query find its best contexts. here we will consider top 10 contexts from dataframe of over 300k absctracts for each query
def searchContext(query,bm25, df_new, num):
  """
  Return top `num` results that better match the query
  """
  # obtain scores
  search_terms = preprocess(query) 
  doc_scores = bm25.get_scores(search_terms)
        
  # sort by scores
  ind = np.argsort(doc_scores)[::-1][:num] 
        
  # select top results and returns
  results = df_new.iloc[ind][df_new.columns]
  results['score'] = doc_scores[ind]
  results = results[results.score > 0]
  return results.reset_index()

# given a questions and corresponding contexts, find the answer.
def AnswerQuestions(question, context):
  # anser question given question and context
  encoded_dict = tokenizer.encode_plus(
                        question, context,
                        add_special_tokens = True,
                        max_length = 256,
                        pad_to_max_length = True,
                        return_tensors = 'pt'
                   )
    
  input_ids = encoded_dict['input_ids'].to(torch_device)
  token_type_ids = encoded_dict['token_type_ids'].to(torch_device)
    
  start_scores, end_scores = model(input_ids, token_type_ids=token_type_ids)

  all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
  start_index = torch.argmax(start_scores)
  end_index = torch.argmax(end_scores)
    
  answer = tokenizer.convert_tokens_to_string(all_tokens[start_index:end_index+1])
  answer = answer.replace('[CLS]', '')
  return answer

bm25, df = init(data)
num_contexts = 10
def obtain_all_contexts(query,num):
  context_df = searchContext(query,bm25,df, num_contexts)
  return context_df.abstract.to_list()

def obtain_all_answers(query, all_contexts):
  all_Answers = []
  for context in all_contexts:
    all_Answers.append(AnswerQuestions(query,context))
  return all_Answers

def getResult(query):
  all_contexts = obtain_all_contexts(query,num_contexts)
  all_answers = obtain_all_answers(query,all_contexts)
  return all_answers


def answers(tasks):
  ALL_ANSWERS = []
  All_Questions = []
  for task in tasks:
    for question in task:
      Answers = getResult(question)
      Answers = list(set(Answers))
      ALL_ANSWERS.append(Answers)
      All_Questions.append(question)
      print('Query:',question)
      print('Answers: ')
      for i, ans in enumerate(Answers):
        i = i+1
        print(str(i)+')',ans)
      print('\n********************************************************************************\n')

# questions to be annswered

task1 = [
         "Is the virus transmitted by aerisol, droplets, food, close contact, fecal matter, or water?",
         "How long is the incubation period for the virus?",
         "Can the virus be transmitted asymptomatically or during the incubation period?",
         "How does weather, heat, and humidity affect the tramsmission of 2019-nCoV?",
         "How long can the 2019-nCoV virus remain viable on common surfaces?"
        ]
    
task2 =     [
                  "What risk factors contribute to the severity of 2019-nCoV?",
                  "How does hypertension affect patients?",
                  "How does heart disease affect patients?",
                  "How does copd affect patients?",
                  "How does smoking affect patients?",
                  "How does pregnancy affect patients?",
                  "What is the fatality rate of 2019-nCoV?",
                  "What public health policies prevent or control the spread of 2019-nCoV?"
              ]

task3 =      [
                  "Can animals transmit 2019-nCoV?",
                  "What animal did 2019-nCoV come from?",
                  "What real-time genomic tracking tools exist?",
                  "What geographic variations are there in the genome of 2019-nCoV?",
                  "What effors are being done in asia to prevent further outbreaks?"
              ]

task4 =       [
                  "What drugs or therapies are being investigated?",
                  "Are anti-inflammatory drugs recommended?"
              ]

task5 =       [
                  "Which non-pharmaceutical interventions limit tramsission?",
                  "What are most important barriers to compliance?"
              ]

task6 =       [
                  "How does extracorporeal membrane oxygenation affect 2019-nCoV patients?",
                  "What telemedicine and cybercare methods are most effective?",
                  "How is artificial intelligence being used in real time health delivery?",
                  "What adjunctive or supportive methods can help patients?"
              ]

task7 =       [
                  "What diagnostic tests (tools) exist or are being developed to detect 2019-nCoV?"
              ]

task8 =       [
                  "What is the immune system response to 2019-nCoV?",
                  "Can personal protective equipment prevent the transmission of 2019-nCoV?",
                  "Can 2019-nCoV infect patients a second time?"
              ]

tasks = [task1, task2, task3, task4, task5, task6, task7, task8]
result = answers(tasks)   # prints answersfor all questions above

# for single query
query = "Is the virus transmitted by aerisol, droplets, food, close contact, fecal matter, or water?",
res = getResult(query)