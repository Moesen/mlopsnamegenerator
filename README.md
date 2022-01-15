# MLOpsNameGenerator
## Overall Goal
The goal of the project is to develop a model that is capable of creating Pokémon names based on its description, using principles orginization and version control, reproduceability, etc.

## Framework
The framework we use is [**Transformer**](https://github.com/huggingface/transformers). We intend to use the Natural Language Processing (NLP) part of the framework. The model we are going to use is [GPT-2](https://huggingface.co/gpt2?text=A+long+time+ago%2C+) doing finetuning over it so we can specialize it over our precise problem.

## Data
Initially, we pretend to use the description of each Pokémon using the [PokéAPI](https://pokeapi.co/), which is a RESTful API linked to a database of details of Pokémon.

Relevant querys to the API:

 - Obtain the list of all Pokémon:

    ```
    https://pokeapi.co/api/v2/pokedex/national
    ```

 - Get the description of each Pokémon:
 
    ```
    https://pokeapi.co/api/v2/pokemon-species/{PKMN_SPECIE_NUMBER}
    ```

## Commands

 - `make requirements`: Installs all requirements from `requirements.txt`.
 - `make devrequirements`: Installs additional dependencies for development.
 - `make datafolders`: Creates folders for the data in the project (`data/raw`, `data/processed`, `data/external` and `data/interim`)
 - `make data`: Downloads and process the data.
 - `make clean`: Deletes compiled Python files
 - `make train`: Trains model
 - `make deploy`: Uploads the updates cleaning and fixing style


## RoadMap
### Week 1
The goal of this week is to setup the project. This includes: Setting up the makefile, setting up the first model and a script for training the model, fetching the data required to train the models, setting up hydra to test with hyperparameters and setting up docker for containerization.

|Alba|Alejandro|Gustav|
|-|-|-|
|Data obtaining and processing|Test usage of GPT-2|Develop model using GPT-2|
|Hydra and config. files|Review and change structure of the train script| Docker images and containers |
|Add wandb to log training progress|Do predict script| Implement dvc with Google Drive |

### Week 2
The goal of this week is to continue working on the project, implementing unit tests, continuous integration and use Google Cloud Platform (gcp).

|Alba|Alejandro|Gustav|
|-|-|-|
| GitHub actions | Unit tests for data and model construction | |
| Create gcp project | Changed prediction to generalize it |  |
| Setup dvc in gcp storage | Calculate coverage ||

### Week3

### Docker commands
Run train:
 - `sudo docker build -f trainer.dockerfile . -t trainer:latest
 - `sudo docker run --name $CONTAINERNAME -v $(pwd)/models/:/app/models/ trainer:latest`
Run predict:
 - `sudo docker build -f predict.dockerfile . -t predict:latest
 - `sudo docker run --name $CONTAINERNAME `


## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── .dvcignore
    ├── .gitignore
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   |    ├── test.csv
    │   |    ├── train.csv
    │   |    └── val.csv
    │   └── raw            <- The original, immutable data dump.
    │        └── raw_descriptions.csv
    │
    ├── data.dvc
    |
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    ├── outputs            <- Training outputs from hydra
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    ├── requirements_devel.txt <- The requirements file for the project development
    ├── requirements_tests.txt <- The requirements file for testing the project
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── download_data.py
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   ├── architectures <- Model architectures
    │   │   ├── configs       <- Yamls of experiments
    │   │   ├── predictions   <- Input and output of predictions
    │   │   ├── predict.yaml
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    ├── tests              <- Test of the project
    │   ├── test_data.py
    │   ├── test_model.py
    │   └── test_predict.py         
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

## Cites and references
[PokéAPI](https://pokeapi.co/)

[Movie name generation with GPT-2](https://www.nbshare.io/notebook/976197999/Movie-Name-Generation-Using-GPT-2/)

[Huggingface transformers](https://github.com/huggingface/transformers)

[Huggingface notebooks](https://github.com/huggingface/notebooks/)

[NameKrea An AI That Generates Domain Names](https://github.com/cderinbogaz/namekrea)


--------
<p><small>DTU Course 02476 - Machine Learning Operations</small></><br>
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
