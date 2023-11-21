# CapstoneProj
Project title:  “Cellular Network Drive Test Log Analysis and Classification”
Team members are:
1. Engr Muhammad Raza  2.Engr Aitizaz Haider  3. Engr Abdul Wahid
# INTRODUCTION
The purpose of this project is to automate the report generation during PTA QoS Survey, will minimize time duration as well as human errors in manual processing of different drive test logs based on different features to optimize cellular network
During drive testing of cellular network, multiple logs files of format “23Feb09 121917.1.nmfs” and “23Feb09 121917.2.nmfs” are being collected, analysize and optimize network performance on bases of these log files in specialized and licensed software Nemo and TEMs.
#  DATASET DESCRIPTION
The dataset used in the project is drive test logs, collected during drive testing at different clusters in GB area (Gilgit, Hunza, Sost, Gahkuch, Astore, Chilas, Skardu, Khaplu and Shigar).
The source of data is Nemo Outdoor software collected logs during drive testing at different clusters in GB.
The total number of logs collected are five hundred (550) and size of dataset is 900 MB.
The CSV file containing useful features such as “log name”, “date”, “time”, “size of log”, “label” and “Slice” (a part of log name) has been formulated from collected drive test logs
# DATASET IMPORTANT FEATURES
Total 500 logs are collected drive test logs data and a CSV file is generated, containing following features:
 “log name”, “label”, “date”, “time”, “size of logs in KB” and “Slice”
From generated CSV file data, two features “Size of Logs” and “Slice” (a part of log name) are important and will be considered as input and log category i.e. “Label” will be considered as output for selected supervised machine learning model.
# BEFORE DATA OVERSAMPLING
The number of samples in all four classes are not same, therefore the performance of model affected and low accuracy has been observed
# AFTER DATA OVERSAMPLING
The number of samples in all four classes are balanced by oversampling and therefore the performance of model has been enhanced and high performance has been achieved
# PROBLEM STATEMENT
Cellular network logs collected during drive testing can’t be identify and categorized easily as there are a lots of logs (550 logs).
The collected drive testing logs may be categorized and labelled manually which will consumed a lot of time (even days) as well as human error for formulation during final report generation.
By this project, the drive test logs will be categorized automatically which will reduced time (in Minutes) as do during manual manipulation for final report generation in PTA QoS activities.
# PROPOSED METHODOLOGY - STEPS
Following steps have been carried out during capstone project:
Data Collection: Data as log files were collected by our selfs during drive testing of cellular network at different clusters(cities) 
Data Preprocessing:Data has been preprocessed and a CSV file from log files has been fomulated. Moreover, data outliers and null values are been observed(no null values and outliers)
Data Visualization: Then the collected data has been visualized by using line graphs, bar charts, histrogram and heat map to understand insight of data and identify relationship between selected features.
Data Encoding: Label encoding has been performed on the output class of our data.
Data Balancing: As the output classes of data have imbalance number of data points, therefore datahas been balanced by using SMOTE liberary function to achieve greater accuracy and performance metrics. 
Data Splitting: Data has been split into train and testing data for training and evaluation of model 
Model Selection: As our data has input as well as output labels therefore supervised machine learning model Random Forrest and Deep Leaning Model ANN have been implemented, trained and output performance metrics (accuracy, precision, recall and F1 – score) has been evaluated
Model Evaluation:The performance of above implemented supervised machine learning and deep learning has been evaluated graphically as Confusion Matrix and ROC
# CONCLUSION
The performance metrics are found quite accurate and the log file are classified into desired labels (mobile terminating, mobile originating, 3g data and 4g data)
The output classses are being further included in file report showing perfomance of cellular network
The lacks and drawbacks of cellular network then eliminate by different actions  including antenna adjustment, addition of new antennas, addition of new site to improve data throughputs, enhanced coverage etc  
This project is very good in atomation of cellular network drive test logs and final report generation containing cellular network performance



