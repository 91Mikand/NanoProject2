# Disaster Response Pipeline Project
This project is a part of Udacity's Data Scientist NanoDegree.

# Table of Contents

## 1. [Project Motivation](#motivation)
## 2. [Installations](#installations)
## 3. [Project Overview](#overview)
## 4. [Project Folders and Files](#files)
## 5. [Acknowledgments](#ack)

<a id='motivation'></a>
## 1. Project Motivation

The objective of the project is to create a web app that classifies text messages about disasters. The project has been created with cooperation with Figure Eight who provided the datasets.
The practical use of the web app would be for the authorities to react in a swift manner after receiving a message about a disaster. 
When a user enters the message, it gets classified and reaches all the departments which should be contacted during this kind of a disaster.

<a id='installations'></a>
## 2. Installations

The project requires the following Python libraries:

* pandas
* json
* plotly
* re
* sys
* sklearn
* nltk
* sqlalchemy
* pickle
* Flask
* sqlite3


<a id='overview'></a>
## 3. Overview

### 1.	ETL Pipeline

The first script created for this project extracts, cleans and loads the data using the Extract-Transform-Load process. 
In this step, the raw datasets provided by Figure Eight get transformed in a way that allows the app to classify the messages more easily. The data is stored in a sql database.

### 2.	Machine Learning Pipeline

In the next step it prepares a Machine Learning pipeline (ML) by extracting the same data saved in the previous step in SQL database. In Machine Learning part, it loads data, apply Natural Language Processing (NLP) steps, build Machine Learning model, fine tune model and then evaluate model. Finally, store the model in a pickle file.

### 3.	Flask Web App

In the final step of this project, we have to build a Flask web app where a user enters the disaster message and gets the message classifications based on the entered text.


<a id='files'></a>
## 4.	Project Files and Folders
This project contains several folders and files
1. The app folder contains the folder called templates and a script called  run.py. The script runs the web app. In the templates folder there are two html files - the front end of this site consists of those two files.
2. The data folder contains two datasets used to build the model ("disaster_categories.csv" and "disaster_messages.csv"), the ETL script ("process_data.py") and the database created with said script ("DisasterResponse.db"
3. The models folder contains a script that creates, trains, evaluates and saves the model ("train_classifier.py")
4. The main folder contains a README.md file with instructions how to you the model and the saved model as a .sav file ("model.sav)


<a id='ack'></a>
## 5. Acknowledgments
Thank you Figure Eight and Udacity for this amazing opportunity and help. While doing the project, I found all the answers to my questions in the Knowledge section on Udacity.
