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
4. Updated ml/model.py.def_train with HyperParameter Tuning using GridSearchCV
5. Updated ml/model.py.def_train with RandomForestClassifier 
6. Updated ml/model.py.def.inference with preds = model.predict(X) 
7. Updated ml/model.save w import joblib for save/load model functions in ml/model.py
8. Updated ml/model.py.save_model with joblib.dump(model, path) to save model
9. Updated ml/model.py.load_model with return joblib.load(path) to load model.
10. Added code to ml/model.def.performance_on_categorical_slice to computer model metrics. 
11. Added three test units in test_ml.py
    a. Test that train_model returns a RandomForestClassifier with a predict method
    b. Test to ensure inference returns predictions matching input length.
    c. Test that performance_on_categorical_slice returns precision, recall, fbeta values.
12. Ran all unit tests and uploaded screenshot UnitTestsPassed.jpg to screenshotfolder.
13. Added path for encoder, scaler, and model in main.py
14. Added FastAPI code in main.py
15. Added Welcome Message in main.py
16. Updated data_processed with necessary code.  
17. Ran Train_Save and saved screenshot to directory.  
18. Added Post/Get screenshots to directory
19. Updated local_api.py, ran and added screenshot to directory. 
20. Updated model_cad.md with necessary information. 
21. Useful links
    GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    HyperParameter Tuning: https://scikit-learn.org/stable/modules/grid_search.html
    Model_Cards: https://www.kaggle.com/code/var0101/model-cards
    RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    Unit Testing in Python: https://realpython.com/python-testing/
    Encoding Categorical Data in Sklearn: https://www.geeksforgeeks.org/encoding-categorical-data-in-sklearn/
    train-test-split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html


