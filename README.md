# Disaster Response Pipeline Project 

## Project overview

![Figure eight logo](https://upload.wikimedia.org/wikipedia/en/a/a6/Attached_to_figure-eight-dot-com.png)

Managing Disasters by quickly addressing the issue and providing the need help and support, requires an near immediate sharing of the given issue. If people can get their help quickly just by analyzing the arrival of disaster related messages and by quickly parsing them to identify the type of issue and what needs to be done ? Here comes Figure Eight.

```
Figure Eight (formerly known as Dolores Lab, CrowdFlower) is a human-in-the-loop machine learning and artificial intelligence company based in San Francisco.
It uses human intelligence to do simple tasks such as transcribing text or annotating images to train machine learning algorithms.

```

This project is about analyzing disaster data from Figure Eight to build a model for an API that classifies disaster messages. It helps to parse and preprocess csv raw data, input it into a machine learning pipeline and output a predicted type of message . The obtained result can help to analyse messages and to identify the specific type of issue so that it can be addressed right away .

The project is structured in two main parts : 

* **ETL and ML pipelines** that take cares of importing, preprocessing and pertaining a model for later usage
* **An app that runs on FLASK** and that help to visualize the results of the whole process by using some plotly figures

Check the further details section to go deeper into the prerequisites and how to run the app.

## Prerequisites and Instructions

### Dependencies

To run the project app and pipeline scripts, the user needs to intall the folowing items : 

* **Python 3.6x**
* **Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn**
* **Natural Language Process Libraries: NLTK**
* **SQLlite Database Libraries: SQLalchemy**
* **Model Loading and Saving Library: Pickle**
* **Web App and Data Visualization: Flask, Plotly**

### How to:

* **Downloading** : To download and use the project app, clone the repo by using this command : 

  * `git clone https://github.com/ElMedBen/figure8.git`

* **Running** : To use the app you will need to run the following commands : 

  * **ETL Pipeline** : To prepare the downloaded csv from figure 8 and to save it into a database, you can run the following command `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`

  * **ML Pipeline** : To build and export a pretrained model as a pkl file for later use, you can run the following command `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

  * **Run app** : To run the app and use the classification pkl for predicting and showing results, you can run the following command `python app/run.py` and to check the result, just open this link `http://localhost:3001` on your browser.

## Folder structure :
```
├─ app
│  ├─ run.py : The main app that you run to open the ip on your browser
│  └─ templates
│     ├─ go.html : Html file containing the updated list within table that shows what type of messages were predicted
│     └─ master.html : The main html file containing the structure of the of the app and that call in the values from our scripts.
├─ data
│  ├─ DisasterResponse.db : Example already provided of a database created after preprocessing of data
│  ├─ disaster_categories.csv : Example of categories in csv format given for prediction
│  ├─ disaster_messages.csv : Example of messages inc csv format given for analysis
│  └─ process_data.py : Script that perform as a simple ETL to prepare the data
├─ models
│  ├─ classifier.pkl : Example of dumped model that can be later use for classification within the app
│  └─ train_classifier.py : Training script that perform as a simple ML pipeline to generate the pretrained model
└─ README.md
```

## Author
[El Mehdi Benammi](https://github.com/ElMedBen)

## License 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)





