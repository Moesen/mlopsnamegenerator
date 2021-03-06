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
 - `make pickle`: Pickle a model and a tokenizer


### Docker commands
Run train:
 - `sudo docker build -f trainer.dockerfile . -t trainer:latest`
 - `sudo docker run --name $CONTAINERNAME -v $(pwd)/models/:/app/models/ trainer:latest`

Run predict:
 - `sudo docker build -f predict.dockerfile . -t predict:latest`
 - `sudo docker run --name $CONTAINERNAME`


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
| GitHub actions | Unit tests for data and model construction | GCP docker images |
| Create gcp project | Changed prediction to generalize it | Parallel data fetching |
| Setup dvc in gcp storage | Calculate coverage | |

### Week3
The goal of this week is to deploy the model and prepare the hand in of the project.

|Alba|Alejandro|Gustav|
|-|-|-|
| Deploy model locally with Torchserve | Deploy model using gcp functions | Build trainer container with gcp using Cloud Build Triggers|
| Work on the presentation | Work on the presentation | Work on the presentation |

## Course Checklist

#### S1 - Getting Started
- [x] Conda
    - Each member uses a `conda` environment 
- [x] PyTorch
    - The develop and training of the model is in `PyTorch`

#### S2 - Organization and version control 
- [x] Git
    - Link to [GitHub repository](https://github.com/Moesen/mlopsnamegenerator)
- [x] Code structure
    - We use `cookie-cutter` basic structure
- [x] Good coding practice
    - We use docstrings in all our methods
    - For styling we check `flake8` and use `black` and `isort`
    - We use explicit typing in our methods definition
- [x] Data Version Control 
    - For our data, we use `dvc`

#### S3 - Reproduceability 
- [x] Docker
    - We have a docker image for training and another for prediction
- [x] Config files
    - With `hydra` we load the configuration of the model with `.yaml` files

#### S4 - Debugging, profilling and logging 
- [ ] Debugging
    - It was not needed to do a complex debugging
- [ ] Profilling
    - As we use pretrained models, and API for the training loop, we did not require any profilling tool
- [x] Experiment logging
    - Register each experiment with `wandb`. We also use `hydra` for this
- [x] Minimizing boilerplate
    - Using Huggingface Trainer API for `torch`

#### S5 - Continuous X 
- [x] Continuous Integration
    - We set multiple unit tests with `pytest`
    - We set two GitHub actions for styling and testing
- [ ] Continuous Machine Learning

#### S6 - The cloud 
- [x] Cloud setup
    - We created a Google Cloud shared project
- [x] Using the cloud
    - We created a data storage merged with `dvc`

#### S7 - Scalable applications 
- [x] Distributed data loding
    - Parallel calls to the data API
- [ ] Distributed training
- [x] Scalable inference
    - We decided to use `gpt2` instead of `gpt2-medium`, `gpt2-large` or `gpt2-xl`
    - Possibility of using `fp16` for quantization in training

#### S8 - Deployment 
- [x] Local deployment
    - Model deployed with custom handler in Torchserve
- [x] Cloud deployment
    - Model deployed with Google Cloud Functions
- [ ] Data drifting

#### S9 - Monitoring 
- [ ] System monitoring


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
