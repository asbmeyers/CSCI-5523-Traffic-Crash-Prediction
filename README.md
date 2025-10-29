# CSCI 5523 Project - Traffic Crash Prediction

## Project Focus:
 
We plan to focus on predicting which city grid cells will have a ≥ 1 traffic crash tomorrow. 

## Motivation: 

This project will allow us to use core data-mining skills (feature selection, imbalanced classification, likelihood/probability assessment) to measure the impact of road-safety, a data-driven problem with clear ground truth (police crash logs).

## Project Plan:

### Data: 

Use Minnesota Crash Data dataset available at the Road Safety Information Center of Minnesota.

### Labelling: 

Discretize the city into fixed grid cells (e.g. 500m). For each day t, label c as 1 if ≥ 1 crash on day t+1, 0 otherwise.

### Features: 

TBD

### Models: 

Baseline: Kernel Density Estimation (KDE).
Supervised: logistic regression, gradient boosting (LightGBM/XGBoost).
Poisson process to calculate the probability of a rare-event occurring within a fixed interval of time or space, given a known average rate of occurrence.

## Milestones:

Use a grid generator to create the city grid cells and assign team roles for the project. 
Evaluate, transform, and load crash/collision dataset for pre-processing
Assign labels 1 for >= 1 crash, 0 otherwise for classification and model training
Identify and select the best features and the parameters for the model
Train the model
Evaluate the model
Create tables and figures to present our findings
Make recommendations e.g. allocating more resources or targeting cells with higher predicted risk, revamping the road design, etc.
Create a final report summarizing everything

## Evaluation Plan:

### Predictive Metrics:

Precision: When we flag high-risk cells, how often it is right.

Recall: How many true hotspots we catch.

### Scoring Rules: 

We will use the error function either Brier (mean squared error of probabilities) or Log-loss function to evaluate accuracy of the probabilities.

### Hit-rate at cell k: 

If we visit k cells tomorrow, how many actually see ≥ 1 crash?

### Hit-rate by area: 

If we cover only N% of the city area, what percent of next-day crashes fall inside?

### Baselines vs Predictions:

Persistence: “tomorrow’s hotspots = today’s hotspots”

KDE: density from recent crashes

Logistic regression: simple supervised baseline

Boosted trees (XGBoost/LightGBM): pick the strongest model.

## Group Members (sign name below)

* Adam Meyers