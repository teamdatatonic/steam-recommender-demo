# Demo 2: Custom Estimator

> The goal of this demo is to implement a custom estimator for a category of models not currently supported by
> the Estimator API.
> This is a demo on live recommendations based on the public Steam dataset that we present at our Machine
> Learning workshops, in collaboration with Google Cloud, to teams of Data Scientists interested in discovering ML.

# Getting Started

This project is developed using Python 3.5.0 and above.

Please, create an environment (if not already present) for the project.

## Installing

Clone the project by running:

```
git clone https://github.com/teamdatatonic/steam-recommender-demo
```

To install the dependencies, first create a virtual environment and, then, run

```
pip install -r requirements.txt
```

To install the repository as a package, so that each file in the submodules
can be run individually, run

```
pip install -e .
```

If more requirements are needed for your R&D, please update the requirements.txt within this repo.

# Package

- `README.md`
- `setup.py`
- `__init__.py`
- `ml_engine_train.sh`- bash script to send out the command for training and hyperparameter tuning to Cloud ML Engine
- `hp_config.yaml` - config file for ml engine hyperparameter tuning
- `deploy.sh` - bash script to deploy a model to Cloud ML Engine for serving
- `online_instance.json` - example JSON file for online prediction
- `batch_instances.json` - example JSON file for batch prediction
- `online_prediction.sh` - bash script to send out the command for requesting an online prediction from deployed model on Cloud ML Engine
- `batch_prediction.sh` - bash script to send out the command for requesting a batch prediction from deployed model on Cloud ML Engine
- `processing/preprocess.py` - script to prepare datasets
- `processing/evaluate.py` - script performing evaluation
- `trainer/__init__.py`
- `trainer/task.py` - training script, which imports from `input.py` and `model.py`
- `trainer/input.py` - script to define how data is served to the model
- `trainer/model.py` - script to define a custom TensorFlow model (factorization machine)


## Custom Estimator

The implemented custom estimator performs a low-rank Matrix Factorization as a second-order inner product of the embedded user and game IDs (first mapped to categorical columns with vocabulary file), and trains by gradient descent. The label is a custom integer rating derived from the playtime (see white paper for more details); unseen interactions have 0 rating when training.

We make another embedding as a bias term to better represent the users' differing average behaviour: e.g., some users may be hard-core movie watchers and may give lower rating, some other users may be loving every movie. The same idea is applied to the items, and also global bias.

## Dataset

The data would be stored in GCS: _gs://example-bucket/_.
Datasets of interest are:

- [x] games.csv -- game vocabulary
- [x] users.csv -- user vocabulary
- [x] played_games/\*.csv -- interactions between each user and game with corresponding rating
- [x] unplayed_games/\*.csv -- unseen interactions between each user and game (with 0 rating)

## Feature Engineering

The Steam dataset is a public dataset that can be downloaded from https://steam.internet.byu.edu/; we have run the SQL server for 2 days to create the csv tables in GCS (not covered here).

To perform the data processing which outputs the train and test sets used for the demo, run

```
python processing/preprocess.py
```

The data pipeline simply selects the feature columns of interest (steamid, appid and rating), and balances the true (played) and false (unplayed) interactions to be in the same number.
If the --is_regression flag is specified, than the playtime is used as the label. Please refer to the dataset at gs://example-bucket/demo2/data/regression.
Else, the rating is creating from percentiles of playtime, thus defining classes 1 - 5. 1 is a relatively low time spent playing the specific game, 5 is in the top 5% of playtime

Running the preprocessing script takes roughly 5 minutes with this downsampled dataset.

## Run

---

The following steps should be performed in order, either locally or on ML Engine.

### Train

---

The model can be trained locally and/or on ML Engine.

#### Locally

```
python train.py [options]
```

Options:

```
	--version=<str>        model version name
    --dim=<k>              dimension of the latent space for factorization [default: *3*]
    --epochs=<n>           number of epochs to train for [default: *1*]
    --lr=<lr>              learning rate while training [default: *0.001*]
    --n_classes=<x>        n classes for rating
    --eval=<boolean>       if perfoming training or just evaluation
    --MODEL_BUCKET=<str>   GCS bucket where to save the model
    --train_data=<str>     GCS bucket where to read train set from
    --test_data=<str>      GCS bucket where to read test set from
```

#### ML Engine

Configure the `hp_config.yaml` and arguments in `ml_engine_train.sh` appropriately.
Remove the hyperparameters object entirely to run a plain one-off training job.
Then run

```
bash ml_engine_train.sh <JOBNAME>
```

to start the ML Engine training job.


### Predict on ML Engine

```
bash online_prediction.sh <model> <version> <file>
```

where <file> is the path to the prediction json file. (e.g. instances.json)<model> is the model name, and <version> is the version name.


## Batch prediction and evaluation

```python3 processing/evaluate.py```

This script extracts a sample of 1000 users, and makes predictions for every game for each of the users. The results are stored on Google cloud storage. Then the games are ranked by their predicted rating for the purposes of evaluating the model.

The ML Engine prediction job evaluating 5 users takes approximately 3 minutes on 1 machine.
The remainder of the script takes a further 3 minutes.
