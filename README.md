DTU - MLOPS - Group 22
==============================
# Link: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview
# Topic: Toxic Comment Classification
# Dataset
1. train.csv : the training set, contains comments with their binary labels
2. test.csv : the test set, you must predict the toxicity probabilities for these comments. To deter hand labeling, the test set contains some comments which are not included in scoring.
3. sample_submission.csv : a sample submission file in the correct format
4. test_labels.csv : labels for the test data; value of -1 indicates it was not used for scoring
# Model
- For the project, DistilBERT is going to be used. DistilBERT is a distilled version of BERT (Bidirectional Encoder Representations from Transformers), which is a popular language model developed by researchers at Google. Compared to original BERT, DistilBERT has fewer parameters. Despite having fewer parameters, DistilBERT is able to achieve good performance on a wide range of natural language processing tasks, including sentiment analysis, question answering, and language translation.
# Chosen framework
- PyTorch is going to be used for this project. General structure would be cookiecutter for organising project's code base as well as docker for cotainerisation, and for configuration, hydra is going to be used. Logging would be shown by using wandb. Lastly, the project would do continious integration by using cloud deployment on Google. 

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
