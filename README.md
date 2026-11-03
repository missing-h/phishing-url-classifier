# Phishing URL Classifier

## Overview

Machine learning-based system developed to identify potentially malicious URLs and distinguish phishing websites from legitimate websites.

This project was developed as my final degree project in Computer Engineering.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- SMOTEENN
- Joblib

## Main Features

- URL feature extraction and preprocessing.
- Identification of suspicious URL patterns.
- Machine learning classification using Random Forest.
- Handling of imbalanced datasets.
- Classification threshold adjustment to address false positives.

## Methodology

The system uses a Random Forest classifier trained on URL-related features. SMOTEENN was incorporated to address class imbalance.

Particular attention was given to reducing false positives, which was one of the main challenges during development.

## Project Structure

- `src/`: Source code.
- `data/`: Dataset documentation.
- `models/`: Model documentation.
- `notebooks/`: Experiments and analysis.

## Project Status

Academic project developed as part of my degree requirements.

## Author

Computer Engineering graduate.
