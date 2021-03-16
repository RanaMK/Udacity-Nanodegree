# Disaster Response Pipeline Project

This project aims at using machine learning to classify messages sent during disaster events into 36 categories. It uses Multiclass classifier to ategorize these events so that the messages are sent to an appropriate disaster relief agency.

The project code is divided into two parts:
1. ETL pipeline to extract data from csv files and load it into a database.
2. Machine learning pipeline for message classification and evaluation.

The project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 


# Project Structure anf Files in the repository

1. app folder -- contains the web app code with the following files:
    - template
        - master.html # main page of web app
        - go.html # classification result page of web app
    - run.py # Flask file that runs app
2. data folder -- contains the data files and the code to process the data with the following files:
    - disaster_categories.csv -- data to process
    - disaster_messages.csv -- data to process
    - process_data.py -- code to extract the data from csv files into python dataframes and make some data cleaningbefore loading it into a database table.
    - InsertDatabaseName.db -- database to save clean data to
3. models folder -- contains the code for machine learning pipeline as well as the saved output model as pickle file with the following files: 
    - train_classifier.py -- code to read the data from the database table and feed it into the Machine learning model to be trained. Then, it contains the code to                             test the model and evaluate it with some testing data. At the end, the model is saved as pickle file.
    - classifier.pkl -- saved model
4. README.md -- Project Overview and Files Structure with instructions to use the code files.

# Instructions to use the code files:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
