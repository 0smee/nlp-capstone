[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/0GBBWOiF)
## Sentiment Analysis of Amazon Reviews
=========================
Archive notebooks show old work no communication in them. I have worked in:
first-explore
preprocessing both single words and ngrams
basic_logistic_ngrams
decision_tree_ngrams
randomforest

I plan to try add more context in the preprocessing stage, and improve the communication in the notebooks




### Project Overview

# Identifying customer painpoints in Amazon Reviews 
#### Capstone project BrainStation - Edo Spigel Emmerich
I will lay out the aims of this capstone project:
1. *How can we use machine learning to better understand the sentiment of reviews?*
2. *How can we use machine learning to understand common user complaints about products?*

By utilising machine learning techniques, I hope to be able to build a system which identifies key factors leading to either a positive or negative review. Due to the class imbalance favoring positive reviews, which is discussed later, it will particularly be interesting to try understand in which ways different models can interpret different types of negative reviews. Being able to extract and track common and new issues surrounding a product is an invaluable tool to a business and can act as a first line of service when identifying customer painpoints. 

I will try and create a system which allows the user to utilise different models in order to uncover the key reasons behind the rating of reviews. This should work both on an individual textual input (1 review), as well as on a larger set of reviews (Later).

There are many services available to perform Sentiment Analysis for businesses, such as [Amazon Comprehend](https://aws.amazon.com/comprehend/) and [Google Cloud Natural Language](https://cloud.google.com/natural-language/docs/analyzing-sentiment), however these solutions are often costly, and being able to perform this analysis *inhouse* is more cost-efficent, as well as giving the business user much more control. 

#### The data:
The data was collated by:\
Justifying recommendations using distantly-labeled reviews and fined-grained aspects\
*Jianmo Ni, Jiacheng Li, Julian McAuley University of California, San Diego*\
Empirical Methods in Natural Language Processing (EMNLP), 2019\
[pdf](https://cseweb.ucsd.edu/~jmcauley/pdfs/emnlp19a.pdf)
### Walkthrough Demo

...
...
...
### Project Flowchart
![flowchart](https://github.com/brainstation-datascience/capstone-project-0smee/blob/main/reports/figures/capstoneflowchart.drawio2.svg?raw=true)
...
...
### Overview of current model performance
| Model                                   | Parameters                                                              | Balanced data?                        | Accuracy | Precision on Negative class | Recall on Negative class | F1-Score on Negative class| Notes                                      |
| --------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------- | -------- | --------- | ------ | -------- | ------------------------------------------ |
| Logistic Regression                     | C=10,penalty='l1', solver='liblinear',random_state=42                   | Unbalanced (76% vs 24%)               | 0.73     | 0.42      | 0.31   | 0.36     |                                            |
| Logistic Regression                     | C=10,penalty='l1', solver='liblinear',random_state=42                   | Balanced (Majority class downsampled) | 0.82     | 0.58      | 0.83   | 0.68     |                                            |
| Decision Tree                           | criterion='entropy', max_depth=25, min_samples_leaf=5, random_state=12 | Unbalanced (76% vs 24%)               | 0.75     | 0.46      | 0.21   | 0.28|
| Decision Tree                           | criterion='entropy', max_depth=40, min_samples_leaf=10,random_state=12 | Balanced (Majority class downsampled) | 0.73     | 0.47      | 0.80   | 0.59     |                                            |
| Random Forest | max_depth=300, n_estimators=60, random_state=42                                        | Unbalanced (76% vs 24%)               | 0.71     | 0.40      | 0.37   | 0.38     |                                            |
| Random Forest | max_depth=600, n_estimators=80, random_state=42                                        | Balanced (Majority class downsampled) | 0.81     | 0.57      | 0.83   | 0.68     |                                            |

![ROC Curves GIF](https://github.com/brainstation-datascience/capstone-project-0smee/blob/ff7d8901998f477f44430eadfc6a524c0d8ab47b/gif/rocgif.gif)


...
...
...

### Project Organization

```
.
├── LICENSE
├── Makefile
├── README.md
├── README.md.save
├── data
│   └── [Google Drive](https://drive.google.com/drive/folders/1vIgwqKBAsJvMQIZZfR_u03OFdF4BsEj5?usp=sharing)
├── environment.yml
├── model
│   └── sentiment-model.pkl
├── notebooks
│   ├── 01-first_explore.ipynb
│   ├── 02-newEDA.ipynb
│   ├── 03-preprocessing.ipynb
│   ├── 04-logistic-model.ipynb
│   ├── 05-decision-tree-model.ipynb
│   ├── 06-randomforest-model.ipynb
│   ├── 07-model-comparison.ipynb
│   ├── 100-modelling.ipynb
│   ├── 101-modelling.ipynb
│   ├── README.md
│   ├── __pycache__
│   ├── archive
│   └── source.py
├── references
│   └── papers.md
├── reports
│   └── figures
└── src
```

```
.
├── LICENSE
├── Makefile
├── README.md
├── README.md.save
├── data
│   └── [Google Drive](https://drive.google.com/drive/folders/1vIgwqKBAsJvMQIZZfR_u03OFdF4BsEj5?usp=sharing)
├── environment.yml
├── gif
├── environment.yml
├── model
│   └── [Google Drive](https://drive.google.com/drive/folders/1MPfcePpCU3atEoUAYmalRZPhgdG-80cg?usp=sharing)
├── notebooks
│   ├── 01-first_explore.ipynb
│   ├── 02-newEDA.ipynb
│   ├── 03-preprocessing.ipynb
│   ├── 04-logistic-model.ipynb
│   ├── 05-decision-tree-model.ipynb
│   ├── 06-randomforest-model.ipynb
│   ├── 07-model-comparison.ipynb
│   ├── README.md
│   ├── __pycache__
│   │   └── source.cpython-38.pyc
│   ├── archive
│   └── source.py
├── references
│   └── papers.md
├── reports
└── src
```


* `data` 

Data folder:
    - https://drive.google.com/drive/folders/1vIgwqKBAsJvMQIZZfR_u03OFdF4BsEj5?usp=sharing

* `model`
    - joblib dump of final model / model object
    - https://drive.google.com/drive/folders/1MPfcePpCU3atEoUAYmalRZPhgdG-80cg?usp=sharing

* `notebooks`
    - contains all final notebooks involved in the project

* `reports`
    - contains final report which summarises the project

* `references`
    - contains papers / tutorials used in the project

* `src`
    - Contains the project source code (refactored from the notebooks)

* `.gitignore`
    - Part of Git, includes files and folders to be ignored by Git version control

* `capstone_env.yml`
    - Conda environment specification

* `Makefile`
    - Automation script for the project

* `README.md`
    - Project landing page (this page)

* `LICENSE`
    - Project license

### Dataset
 ## Data Dictionary

| Column name    | Datatype | Measures| Notes|
| -------------- | -------- | ---------------------------------------------------- | --------------------------------------- |
| overall        | float    | Overall star rating of review   |                                         |
| verified       | boolean  | Whether the review has been verified as real or not. | |
| reviewTime     | object   | Time of review    | If needed can change type to datetime64 |
| reviewerID     | object   | Unique ID of reviewer                                |                                         |
| asin           | object   | Product metadata  |Amazon Standard Identification Number|
| style          | object   | Product metadata                                     |                                         |
| reviewerName   | object   | Name of reviewer                                     |                                         |
| reviewText     | object   | Textual contents of review                           |                                         |
| summary        | object   | Textual summary of review                            |                                         |
| unixReviewTime | int64    | Time of review since Unix Epoch on January 1st, 1970 |                                         |
| vote           | object   | Count of usefulness vote                             |                                         |
| image          | object   | Image of product reviewed                            |                                         |

### Credits & References
Justifying recommendations using distantly-labeled reviews and fined-grained aspects
Jianmo Ni, Jiacheng Li, Julian McAuley
Empirical Methods in Natural Language Processing (EMNLP), 2019 https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

--------
