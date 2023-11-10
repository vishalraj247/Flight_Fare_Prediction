# Flight Fare Prediction

## Overview
Flight Fare Prediction is a sophisticated project aimed at providing accurate fare predictions for flights. This project leverages the strengths of various machine learning and deep learning models, including a Wide and Deep Neural Network, LSTM, XGBRegressor, and Random Forest Regressor each offering unique advantages and insights into the fare estimation process.

## Project Structure
```
FLIGHT_FARE_PREDICTION/
│
├── data/
│ ├── external/ # Data from third party sources.
│ ├── interim/ # Intermediate data that has been transformed.
│ ├── processed/ # The final, canonical data sets for modeling.
│ ├── raw/ # The original, immutable data dump.
│
├── docs/ # Documentation files for the project.
│
├── models/
│ ├── best_model/ # Stored models for deployment.
│ │ ├── best_model_ronik/ # Ronik's best model artifacts.
│ ├── best_model_Shivatmak/ # Shivatmak's LSTM model artifacts.
│ ├── best_model-vishal_raj/ # Vishal Raj's Wide and Deep model artifacts.
│ ├── best_model_aibarna/ # Aibarna's Random Forest Regressor model artifacts.

├── notebooks/ # Jupyter notebooks. Naming convention is a number (for ordering),
│ ├── main_Shivatmak.ipynb # Shivatmak's primary notebook.
│ ├── main_vishal_raj.ipynb # Vishal Raj's primary notebook.
│ ├── notebook_ronik_at3.ipynb # Ronik's exploration notebook.
│
├── references/ # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports/ # Experiment Reports as PDF.
│
├── src/ # Source code for use in this project.
│ ├── data/ # Scripts to download or generate data.
│ │ ├── init.py
│ │ ├── data_preprocessor.py # Data preprocessor script.
│ │ ├── make_dataset.py # Dataset creation script.
│ │ ├── ml_model_data_preprocessor.py # ML model data preprocessing.
│ │
│ ├── features/ # Scripts to turn raw data into features for modeling.
│ │
│ ├── models/ # Scripts to train and evaluate models.
│ │ ├── init.py
│ │ ├── train_model_ronik.py # Training script for Ronik's model.
│ │ ├── train_model_Shivatmak.py # Training script for Shivatmak's model.
│ │ ├── train_model_vishal_raj.py # Training script for Vishal's model.
│ │
│ ├── visualization/ # Scripts to create exploratory and results oriented visualizations.
│ ├── init.py
│ ├── visualize.py
│
├── .gitattributes
├── .gitignore
├── app.py # Streamlit API.
├── LICENSE
├── Makefile # Makefile with commands like make data or make train.
├── README.md # The top-level README for developers using this project.
├── requirements.txt # The requirements file for reproducing the analysis environment.
```
## Installation

To set up the development environment, follow these steps:

1. Clone the repository.
2. Create a virtual environment and activate it.
3. Install the dependencies using `pip install -r requirements.txt`.

## Usage

Instructions on how to use the scripts and run the Streamlit application for flight fare prediction would go here.

## Contributing

Interested in contributing? Check out the contributing guidelines with pre-existing collaborators. Please note that this project adheres to a code of conduct, and by participating in the development of this application, you agree to abide by its terms.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the team members who have contributed to this project:

- Vishal Raj: For deep learning model preprocessing and training the Wide and Deep Neural Network.
- Shivatmak: For training the LSTM model.
- Ronik: For preprocessing data and training the XGBRegressor model.
- Aibarna: For training the Random Forest Regressor model.

We also acknowledge that our successful collaboration overcame the challenges faced while the course of the project.

## Contact

For any queries or further discussion regarding the project, feel free to contact us through the project's issue tracker or directly to the contributors' GitHub Id.
