# Disaster Response Pipeline Project 

## Project overview

![Figure eight logo](https://upload.wikimedia.org/wikipedia/en/a/a6/Attached_to_figure-eight-dot-com.png)

This project is about analysing disaster data from Figure Eight to build a mondel for an API that classifies disaster messages. It helps to parse and preprocess csv raw data, input it into a machine learning pipline and output a predicted type of message .

The project is structured in two main parts : 

* **ETL and ML pipelines** that take cares of importing, preprocessing and pretraining a model for later usage
* **An app that runs on FLASK** and that help to visualise the results of the whole process by using some plotly figures

Check the further details section to go deeper into the prerequisits and how to run the app.

## Prerequisits and Instructions

### Dependencies

To run the project app and pipeline scripts, the user needs to intall the folowing items : 

* **Python 3.6x**
* **Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn**
* **Natural Language Process Libraries: NLTK**
* **SQLlite Database Libraqries: SQLalchemy**
* **Model Loading and Saving Library: Pickle**
* **Web App and Data Visualization: Flask, Plotly**

### How to:

* **Downloading** : To download and use the project app, clone the repo by using this command : 

  * `git clone https://github.com/ElMedBen/figure8.git`

* **Running** : To use the app you will need to run the folowing commands : 

  * **ETL Pipeline** : To prepare the downloaded csv from figure 8 and to save it into a database, you can run the following command `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`

  * **ML Pipeline** : To build and export a pretrained model as a pkl file for later use, you can run the following command `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

  * **Run app** : To run the app and use the classification pkl for predicting and showing results, you can run the following command `python app/run.py` and to check the result, just open this link `http://localhost:3001` on your browser.

## Folder structure :
```
├─ app
│  ├─ run.py
│  └─ templates
│     ├─ go.html
│     └─ master.html
├─ data
│  ├─ DisasterResponse.db
│  ├─ disaster_categories.csv
│  ├─ disaster_messages.csv
│  └─ process_data.py
├─ models
│  ├─ classifier.pkl
│  └─ train_classifier.py
└─ README.md
```

## Author
[El Mehdi Benammi](https://github.com/ElMedBen)

## License 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)





