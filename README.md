# capstone21_annmhuang

Capstone Project using Pre-processed MIMIC-III dataset

## Directory Description

0_raw_data : 
Files are too large even after compression. Folder should contain at least 7 CVSs 
(Pre-processed MIMIC-III tables including admissions.csv, gcs_hourly.csv, icustays.csv, labs_hourly.csv, patients.csv, pt_icu_outcome.csv, vitals_hourly.csv)

utils.py:
Python File with all the utility functions used in jupyter notebooks

1.\ Data\ Extraction\ and\ Preparation.ipynb:
Should be read and used with utils.py file. ETLs are mostly done in utils.py.

1_input.zip:
Compressed 1_input folder. Contains input data file for research question 1 and 2, and for data visualisations

2.1\ (Research\ Question\ 1)\ Build\ and\ Train\ ML\ Model(s).ipynb:
Notebook solely for ML model development

2.2_Research_Question_2_R.Rmd:
R markdown file solely for statistical analysis/modeling for research question 2

2.2_Research_Question_2_R.html:
HTML output of the R markdown file

2_training:
Saved training dataset from ML development notebook

2_validation:
Saved test dataset from ML development notebook

conda_environment.yml:
Conda venv yml file, for environment recreation

Data\ Visalisation\ (of\ Prepared\ ICU\ data\ and\ the\ Final\ Chosen\ Model).ipynb:
Note book for data visualisation and EDA purpose

final_model:
Folder contains the final best performing ML models: GBT and XGB models.

final_reports:
Folder contains the final report in .docx and .pdf format