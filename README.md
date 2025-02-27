# Recommendation System for Amazon Products with enhanced BERT classifier

## Description
We download Amazon product reviews data from https://amazon-reviews-2023.github.io/, load them to a postgres database and develop a recommender system.
- embed reviews using a pre-trained embedding BERT or TF-IDF method
- build a sentiment classifier (compare various models amongst random forests, gradient boosting and BERT)
- create an interactive dashboard for data visualization and model performance analysis
- implement the recommendation system
- wrap up the app using FastAPI routes and dockerinzing it

## Pipeline Snapshot


## Virtual environment
Use the following command lines to create and use venv python package:
```
python3.13 -m venv venv
```
Then use the following to activate the environment:
```
source venv/bin/activate
```
You can now use pip to install any packages you need for the project and run python scripts, usually through a `requirements.txt`:
```
python -m pip install -r requirements.txt
```
When you are finished, you can stop the environment by running:
```
deactivate
```


## Commands
Once you activated environment with necessary requirements installed, you can run the following, from folder level `recommendation_system_bert/`.


Check all commands with
```
python . --help
```

Launch sentiment classification pipelines with
```
python . sentiment {classifier_name} {args}
```



## Interesting Material