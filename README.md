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

I am interested in understanding the reliability of reviews left by customers online. Consumers’ online shopping experience can be improved by better understanding
the reviews that are left on products. By improving the customer experience, online retailers will continue to see a huge increase in total sales. First time buyers will be
particularly impacted by reviews, as they search for their ideal product. Users making large purchases will also be impacted by the quality of reviews available.
Organisations or people following a budget will also use reviews to guide their purchase. We can use machine learning to help better understand the quality of reviews. For example, we can perform sentiment analysis to gain further insights of the text contents of a review, and try to understand how reliable a source the reviewer is. We
can compare the overall sentiment of a review to its “star rating” and try to understand discrepancies in these two different forms of reviews.
### Walkthrough Demo

...
...
...
### Project Flowchart
![flowchart](https://github.com/brainstation-datascience/capstone-project-0smee/blob/main/reports/figures/capstoneflowchart.drawio.svg?raw=true)

...
...
### Overview of current model performance
| Model                                   | Parameters                                                              | Balanced data?                        | Accuracy | Precision | Recall | F1-Score | Notes                                      |
| --------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------- | -------- | --------- | ------ | -------- | ------------------------------------------ |
| Logistic Regression                     | C=10,penalty='l1', solver='liblinear',random_state=42                   | Unbalanced (76% vs 24%)               | 0.72     | 0.71      | 0.79   | 0.70     |                                            |
| Logistic Regression                     | C=10,penalty='l1', solver='liblinear',random_state=42                   | Balanced (Majority class downsampled) | 0.79     | 0.74      | 0.80   | 0.75     |                                            |
| Decision Tree                           | criterion='entropy', max_depth=25, min_samples_leaf=5, random_state=12 | Unbalanced (76% vs 24%)               | 0.79     | 0.71      | 0.64   | 0.66|
| Decision Tree                           | criterion='entropy', max_depth=30, min_samples_leaf=5,random_state=12 | Balanced (Majority class downsampled) | 0.73     | 0.68      | 0.73   | 0.68     |                                            |
| Random Forest (currently not optimised) | n_estimators=30, random_state=42                                        | Unbalanced (76% vs 24%)               | 0.85     | 0.81      | 0.76   | 0.78     |                                            |
| Random Forest (currently not optimised) | n_estimators=30, random_state=42                                        | Balanced (Majority class downsampled) | 0.79     | 0.74      | 0.80   | 0.75     |                                            |

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
│   └── https://drive.google.com/drive/folders/1vIgwqKBAsJvMQIZZfR_u03OFdF4BsEj5?usp=sharing
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

* `data` 

Data folder:
    - https://drive.google.com/drive/folders/1vIgwqKBAsJvMQIZZfR_u03OFdF4BsEj5?usp=sharing

* `model`
    - joblib dump of final model / model object

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
