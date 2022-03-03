# Machine Learning Project 1 - Higgs Boson
Project 1 of the CS433-Machine Learning course given at the EPFL Fall 2021. 

## The team (The Regressionists) is composed by:
- Camillo Nicolò De Sabbata ([@cndesabbata](https://github.com/cndesabbata))
- Gianluca Radi ([@radigianluca](https://github.com/radigianluca))
- Thomas Berkane ([@tberkane](https://github.com/tberkane))

## Project description
The goal of this project is to find the machine learning classification model that would best predict the Higgs boson decay signatures from background noise.
Our best result was achieved using the Ridge regression method, with a categorical accuracy of 0.840 and an F1 score of 0.756 on AIcrowd.

## Structure of the repository: 
- `implementations.py`: contains implementations of Least Squares Regression (Normal, with GD/SGD), Ridge Regression (Normal) and (Regularized) Logistic Regression with GD
- `run.py`: main executable to recreate our best score on AICrowd
- `helpers.py`: some helper functions used by different modules
- `data`: contains the datasets (.gitignore'd)
- `README.md`: this file

## Instructions to run:
Python modules requirements: `numpy`, `matplotlib.pyplot`, `typing` and `csv`. Predictions will be saved in the `data` folder. To reproduce our best score with ridge regression that we submitted on [AIcrowd](https://www.aicrowd.com):
```
python run.py
```
