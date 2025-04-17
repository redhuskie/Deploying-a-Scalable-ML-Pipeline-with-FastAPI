Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up (pip or conda)
* Option 1: use the supplied file `environment.yml` to create a new environment with conda
* Option 2: use the supplied file `requirements.txt` to create a new environment with pip
    
## Repositories
* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.

# Data
* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.

# Project Notes and Updates. 
1. Create python-ci.yaml with Github action. 
2. Updated ML/Data.py with skilearn.scaler.
3. Added Import Pandas and wrote import for data/census.csv in ml/model.py
4. Updated ml/model.py.def_train with HyperParameter Tuning using       GridSearchCV
5. Updated ml/model.py.def_train with RandomForestClassifier 
6. Updated ml/model.py.def.inference with preds = model.predict(X) 
7. Updated ml/model.save w import joblib for save/load model functions in ml/model.py
8. Updated with  joblib.dump(model, path) for saving model
8. Updated ml/model.py.save_model with joblib.dump(model, path) to  save model
9. Updated ml/model.py.load_model with return joblib.load(path) to load model.
10. Added code to ml/model.def.performance_on_categorical_slice to computer model metrics. 
11. Added three test units in test_ml.py
    a. Test that train_model returns a RandomForestClassifier with a predict method
    b. Test to ensure inference returns predictions matching input length.
    c. Test that performance_on_categorical_slice returns precision, recall, fbeta values.