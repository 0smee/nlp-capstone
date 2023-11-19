# Notebook README:
Here I will explain the order/flow of the notebooks:
- `first_explore.ipynb`: initial notebook contianing original EDA, preprocessing and a logistic model.
- `newEDA.ipynb`: Full EDA notebook, other than token frequency analysis which is in `preprocessing_1_2grams.ipynb`
- `preprocessing_1_2grams.ipynb`: Vectorising the reviews, and token frequency analysis
- `basic-logistic_1grams.ipynb`: Logistic model optimisaiton and evaluation on both balanced and unbalanced dataset, all on 1-grams
-`basic-logistic_2grams_real.ipynb`: Logistic model optimisaiton and evaluation on both balanced and unbalanced dataset, all on 2-grams
- `decision-tree_1grams.ipynb`: Decision tree classifier optimisaiton and evaluation on both balanced and unbalanced dataset, all on 1-grams
- `decision-tree_2grams_real.ipynb`: Decision tree classifier optimisaiton and evaluation on both balanced and unbalanced dataset, all on 2-grams
- `randomforest.ipynb`: Random forest classifier, currently not optimised on both balanced and unbalanced dataset, all 1-grams
- `randomforest_2grams.ipynb`: Random forest classifier, currently not optimised on both balanced and unbalanced dataset, all 2-grams
- `source.py`: source file containing all functions which are reused
- `archive`: older unused notebooks
```
.
├── 04-modelling.ipynb
├── 05-findings.ipynb
├── __pycache__
│   └── source.cpython-38.pyc
├── archive
│   ├── basic-logistic-archive.ipynb
│   ├── basic-logistic_2grams.ipynb
│   ├── decision-tree-archive.ipynb
│   ├── decision-tree_2grams.ipynb
│   ├── decision-tree_smote.ipynb
│   ├── preprocessing_2grams.ipynb
│   ├── preprocessing_single-words.ipynb
│   └── spaCy-exploration-one-MISC.ipynb
├── basic-logistic_2grams_real.ipynb
├── basic-logistic_1grams.ipynb
├── decision-tree_2grams_real.ipynb
├── decision-tree_1grams.ipynb
├── first_explore.ipynb
├── model-comparison.ipynb
├── newEDA.ipynb
├── preprocessing_1_2grams.ipynb
├── randomforest.ipynb
├── randomforest_2grams.ipynb
└── source.py
```

