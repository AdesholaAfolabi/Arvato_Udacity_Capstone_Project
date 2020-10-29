# Bertelsmann-Arvato-Project-

This README documents the steps that are necessary to follow through the Bertelsmann/Arvato project. The project has several preprocessing steps which can be found in the preprocessing.py file. Steps like NaN detection, NaN handling, and data engineering are contained in the preprocessing.py python file. The Arvato Project Workbook (a Jupyter Notebook) is where most of the code was executed and the outputs displayed. Lastly, the evaluate_predictions script helps to evaluate the supervised learning algorithm used as demonstrated in the notebook.

The Kaggle_submission3.csv gave the best result on Kaggle and has been included in the project folder as well.


### What is this repository for? ###

* Quick summary: The application employs the modular style of putting the applications together. There is a general module which takes care of data preprocessing (NaN detection and handling, feature engineering), and other preprocessing steps such as One Hot Encoding, Scaling, PCA, etc are done in the Notebook. 
* Version: 1.0
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up? ###

* Summary of set up: ensure you have python 3 up and running
* Configuration: ensure all modules are imported properly. They all depend on each other
* Dependencies: python 3, pandas, scikit-learn, sklearn pre-processing, catboost, xgboost, SHAP, Sagemaker, imblearn
* Database configuration: no required configuration
* How to run tests: no tests files used yet. Version 2 will come with test cases

"README.md" 27L, 1205C
* Repo owner: afolabimkay@gmail.com

