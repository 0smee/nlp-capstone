# source file for functions I reuse
# imports
import pandas as pd
import numpy as np
import gzip
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer  



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
    
    
