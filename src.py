'''
Used Ressources :
- https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/#13viewthetopicsinldamodel
- https://www.kaggle.com/ragnisah/text-data-cleaning-tweets-analysis
'''

import pandas as pd
import numpy as np
import nltk 
import string
import re
import os
import ast
import gensim
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from scipy.spatial.distance import euclidean

os.environ.update({'MALLET_HOME':os.path.join(os.getcwd(), 'other\mallet-2.0.8')})

nltk.download('stopwords')
nltk.download('wordnet')

# Data Loading
def load_data(url):
    data = pd.read_csv(url,)
    return data

# Punctuation removal
def remove_punct(text):
    text  = "".join([char for char in str(text) if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

# Sub division of strings into arrays of words
def tokenization(text):
    text = re.split('\W+', text)
    return text

# Stop words removal 
def remove_stopwords(text):
    stopword = nltk.corpus.stopwords.words('english')
    text = [word for word in text if word not in stopword]
    return text

# Getting the root/base of words
def stemming(text):
    ps = nltk.PorterStemmer()
    text = [ps.stem(word) for word in text]
    return text

# Linking similar words to one word
def lemmatizer(text):
    wn = nltk.WordNetLemmatizer()
    text = [wn.lemmatize(word) for word in text]
    return text

# Text processing
def process_dataframe(data):
    df  = pd.DataFrame(data[['text']])
    df['text_punct'] = df['text'].apply(lambda x: remove_punct(x))
    df['text_tokenized'] = df['text_punct'].apply(lambda x: tokenization(x.lower()))
    df['text_nonstop'] = df['text_tokenized'].apply(lambda x: remove_stopwords(x))
    df['text_stemmed'] = df['text_nonstop'].apply(lambda x: stemming(x))
    df['text_lemmatized'] = df['text_nonstop'].apply(lambda x: lemmatizer(x))
    df = df.drop(columns=['text_punct', 'text_tokenized', 'text_nonstop', 'text_stemmed'])
    df = df.rename(columns={'text':'original_text', 'text_lemmatized':'processed_text'})
    return df

# Get Dominant Topic for text
def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

# Getting Quality Measures of the LDA Model
def getPerplexityCoherence(model, corpus, texts, dictionary):
    perplexity = model.log_perplexity(corpus)
    coherence_model_lda = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model_lda.get_coherence()
    return perplexity,coherence

# Getting dominant topic for each text
def getDominantTopicsForTexts(ldaModel, corpus, texts):
    print('Getting Dominant Topics...')
    df_topic_sents_keywords = format_topics_sentences(ldaModel=ldaModel, corpus=corpus, texts=texts)
    #Formating The Results
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    return df_dominant_topic

# Getting Array of Topics
def getTopics(ldaModel, n_topics):
    print("Retreiving Topics...")
    topics = []
    for index in range(20):
        topic = []
        for word, n in topic_modeler.show_topic(index):
            topic.append(word)
        topics.append(topic)
    return topics

# Training Or loading topic modeler
def getTopicExtractor(corpus, dictionary, load=True):
    if load:
        # Loading Mallet LDA Model
        print('Loading LDA Mallet Model...')
        ldamallet = gensim.models.wrappers.LdaMallet.load("model/mallet_model/mallet")
        return ldamallet
    else:
        # Training Mallet LDA Model
        print('Training LDA Mallet Model')
        mallet_path = os.path.join(os.getcwd(), 'other\\mallet-2.0.8\\bin\\mallet')
        ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=bow_corpus, num_topics=20, id2word=dictionary)
        # Saving The Model
        ldamallet.save("model/mallet_model/mallet")
        return ldamallet

# Training Doc2Vec Model
def trainDoc2VecModel(texts, load=True):
    if load:
        print('Loading Doc2Vec Model...')
        model = Doc2Vec.load("model/doc2vec_model/doc2vec")
        return model
    else:
        print('Training Doc2Vec Model...')
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
        model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
        model.save("model/doc2vec_model/doc2vec")
        return model

# Loading dataset
print('Loading Data...')
news = load_data('./data/20_newsgroup_train.csv')

# Processing dataset
print('Pre-processing Data...')
processed_df = process_dataframe(news)

# Creating dictionnary from text
print("Creating Dictionnary...")
dictionary = gensim.corpora.Dictionary(processed_df['processed_text'])
# Removing useless words
dictionary.filter_extremes(no_below=20, no_above=0.5)

# Transforming text into bag of words
print("Creating bag of words...")
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_df['processed_text']]
# Making it human readable
# id_words = [[(dictionary[id], count) for id, count in line] for line in bow_corpus]

# Getting topic modeler
topic_modeler = getTopicExtractor(bow_corpus, dictionary)

# Getting The Topics
topics = getTopics(topic_modeler, 20)

# Training Doc2Vec on raw text data
doc2VecModel = trainDoc2VecModel(processed_df['original_text'])

# Applying Doc2Vec on Topics
print("Applying Doc2Vec Model on topics")
topic_vectors = [doc2VecModel.infer_vector(topic) for topic in topics]

# Applying Doc2Vec on each document
print("Applying Doc2Vec Model on documents")
document_vectors = []
for doc in processed_df['processed_text']:
    document_vectors.append(doc2VecModel.infer_vector(doc))

# Calculating Similarity
print("Calculating Similarity")
similarity = np.empty((len(document_vectors), len(topic_vectors)))
for i, document in enumerate(document_vectors):
    for j, topic in enumerate(topic_vectors):
        similarity[i,j] = euclidean(document, topic)