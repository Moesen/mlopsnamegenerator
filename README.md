# MLOpsNameGenerator
## Overall Goal
The goal of the project is to develop a model that is capable of creating Pokémon names based on its description, using principles orginization and version control, reproduceability, etc.

## Framework
The framework we use is [**Transformer**](https://github.com/huggingface/transformers). We intend to use the Natural Language Processing (NLP) part of the framework. The model we are going to use is [GPT-2](https://huggingface.co/gpt2?text=A+long+time+ago%2C+) doing finetuning over it so we can specialize it over our precise problem.

## Data
Initially, we pretend to use the description of each Pokémon using the [PokéAPI](https://pokeapi.co/), which is a RESTful API linked to a database of 


## RoadMap
### Week 1
Goal of this week is to setup the project. This includes: Setting up the makefile, setting up the first model and a script for training the model, fetching the data required to train the models, setup hydra to test with hyperparameters and setup docker for containerisation.

|Alba|Alejandro|Gustav|
|-|-|-|
|t|t|t|

### Week 2


### Week3


## Project Organization


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

## Cites and references
[Movie name generation with GPT-2](https://www.nbshare.io/notebook/976197999/Movie-Name-Generation-Using-GPT-2/)

--------
<p><small>DTU Course 02476 - Machine Learning Operations</small></><br>
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
