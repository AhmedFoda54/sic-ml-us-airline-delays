# US Airline Delays Prediction

This project focuses on building a machine learning models to predict delays in US airline flights. The dataset contains information such as departure and arrival times, weather conditions, airports locations, airline codes, and flight delay statuses. The aim is to provide insights and predictions to help airlines and passengers better understand and manage potential flight delays.

## Features
- **Flight Delay Prediction**: Predicts whether a flight will be delayed based on flight and weather data.
- **Arrival and Departure Delays Prediction**: Predicts how much time in minutes a flight will be delayed in the departure or arrival.
- **Data Visualization**: Provides visual insights into the data, showing trends in delays across different airlines, airports, and seasons.

## Project Structure
The project is organized as follows:
- `notebook/`: Jupyter notebooks for data exploration, feature engineering, and model development.
- `models/`: Contains saved models that can be used for inference.
- `app_images/`: Images used in the streamlit web app
- `streamlit_app.py`: Web app for delay prediction, built with Streamlit.

## Dataset
The dataset used in this project includes flight records from various US airlines. The features include:
- **Airline**: The airline operating the flight.
- **Flight Number**: The flight number assigned to the flight.
- **Origin and Destination**: The departure and arrival airports.
- **Departure and Arrival Times**: Scheduled and actual times of departure and arrival.
- **Delay Status**: Whether the flight was delayed, on time, or canceled.
- **Geographical Locations**: Longitude and Latitude for airports.
- **Weather Confitions**: rain, snow, min, max and avg temperature.

The data is sourced from public flight information databases such as the US Department of Transportation and weather-related data from external sources.
You can find the dataset on Kaggle: [US 2023 Civil Flights, delays, meteo and aircrafts](https://www.kaggle.com/datasets/bordanova/2023-us-civil-flights-delay-meteo-and-aircraft)

## Notebook
This is our Kaggle Notebook, you can have a look for more detailed work: [Kaggle Notebook](https://www.kaggle.com/code/ahmedashrafhelmi/us-airline-2023)

## Installation

### Prerequisites
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib
- Seaborn
- Plotly
- Streamlit for web app deployment

### Install Dependencies
1. Clone the repository:
    ```bash
    git clone https://github.com/AhmAshraf1/sic-ml-us-airline-delays.git
    cd sic-ml-us-airline-delays
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data Preprocessing
1. Merging data csv files of locations and weather with the main csv file
2. Prepare the dataset:
    - Clean the data by removing missing or invalid entries.
    - Feature engineering: Add relevant features such as weather conditions or peak travel periods.

### Streamlit Web App
To run the Streamlit app for delay prediction, execute:
```bash
streamlit run streamlit_app.py
```

You can use the Web App dedployed on Streamlit using this link: [Streamlit Web App](https://us-airline-flight-delay.streamlit.app/)

## Visualization
The project provides various visualizations to help understand flight delays, including:
- Delay trends by airline, airport, and time of year.
- Correlation between weather conditions and delays.
- Insights into peak hours and how they affect delays.

## Model
The trained models are saved in the models/ directory and some of machine learning models used include:
- Logistic Regression
- Gradient Boosting
- LightGBM Regressor
- XGBoost Classifier

The performance of each model is evaluated using metrics such as accuracy, precision, recall, and F1-score. The best-performing model is then used for prediction. 

## Future Improvements
- Integrate real-time weather data to improve the accuracy of predictions.
- Expand the model to predict delays in international flights.
- Develop an API for real-time prediction integration with airline systems.
 
