# Disaster Response Pipeline Project

### Instructions:

* **Downloading** : To download and use the project app, clone the repo by using this command : 

  * `git clone https://github.com/ElMedBen/figure8.git`

* **Running** : To use the app you will need to run the folowing commands : 

  * **ETL Pipeline** : To prepare the downloaded csv from figure 8 and to save it into a database, you can run the following command `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`

  * **ML Pipeline** : To build and export a pretrained model as a pkl file for later use, you can run the following command `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

  * **Run app** : To run the app and use the classification pkl for predicting and showing results, you can run the following command `python app/run.py` and to check the result, just open this link `http://localhost:3001` on your browser.



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
