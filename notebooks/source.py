# source file for functions I reuse
# imports
import pandas as pd
import numpy as np
import gzip
import json
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords



# These functions based on https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)  # Using json.loads instead of eval()

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

# functions from the first_explore notebook
def simplifyReviews(dataframe):
    '''
    assumes data from amazon reviews, outputs new dataframe with only verified reviews containing overall rating, review and summary

    '''
    out = dataframe.copy()
    out = out[["overall", "verified", "reviewText", "asin"]]
    out = out[out["verified"]==True]
    out = out.drop("verified", axis=1)
    out.info()
    out = out.dropna()
    out = out.drop_duplicates(ignore_index=True)
    return out
    
def rating_distribution(data):

    values, counts = np.unique(data['overall'], return_counts=True)
    normalized_counts = counts/counts.sum()

    plt.figure()
    plt.bar(values, normalized_counts * 100)
    plt.xlabel("Review rating")
    plt.ylabel('% of reviews')
    plt.title("Rating distribution")
    plt.show()
    return normalized_counts
# bag of words review
def bow_review(review_train,review_test,stop=None, ngrams=(0,1)): 
    # 1. Instantiate 
    bagofwords = CountVectorizer(stop_words=stop, ngram_range=ngrams)

    # 2. Fit 
    bagofwords.fit(review_train)

    # 3. Transform
    small_transformed_train = bagofwords.transform(review_train)
    small_transformed_test = bagofwords.transform(review_test)
    return(bagofwords, small_transformed_train, small_transformed_test)
    
    
def my_tokenizer(document):
    stop_words = stopwords.words('english')
    # remove punctuation
    for punct in string.punctuation:
        document=document.replace(punct,'')

    # tokenize - split on whitespace
    tokenized_document = document.split(' ')

    # pattern denoting a sequence of at least 2 alphanumeric characters
    pattern=r"(?u)\b\w\w+\b"

        # tokenize - split by matching a pattern
    tokenized_document = re.findall(pattern, document)
    
    # remove stopwords before stemming or lemmatization
    tokenized_document = [word for word in tokenized_document if word not in stop_words]


    stemmed_tokens_list = []
    for i in tokenized_document:
        token = PorterStemmer().stem(i)
        stemmed_tokens_list.append(token)
    return stemmed_tokens_list


# function for downsampling X_train - originally from basic-logistic
def downsample_binary(y_t, x_t, mini = 0, maj=1):
    # combine x and y
    data = pd.concat([y_t, x_t], axis=1)
    target_name = data.columns[0]
    # count the instances of the minority class
    minority_count = data[data[target_name] == mini].shape[0]
    # random sample from the majority class
    majority_sample = data[data[target_name] == maj].sample(n=minority_count, random_state=42)
    
    # merge together
    balanced_df = pd.concat([data[data["binary"] == mini], majority_sample])
    
    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df
