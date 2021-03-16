# Disaster Response Pipeline Project

This project aims at using ML to classify messages into 36 categories. It uses Multiclass classifier to achive this goal.

First, we extract the data from csv files into python dataframes and make some data cleaning. Then, this cleaned data is loaded into a database table. (File: data/process_data.py)

In the next step, we read the data from the database table and feed it into the Machine learning model to be trained. Then, we test the model and evaluate it with some testing data. At the end, the model is saved as pickle file. (File: models/train_classifier.py)

In addition, there is a webpage with some simple charts about the data used. (app folder)


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
