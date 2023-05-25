#!/usr/bin/env python
# coding: utf-8

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressionModel
import mlflow.pyfunc
import pandas as pd
from pyspark.ml.linalg import Vectors
import pickle

# Prediction implementation with spark
def spark_predict(vma, acrm):
    # Create a Spark session
    spark = SparkSession.builder \
                .master('local') \
                .appName('StockVolumePrediction') \
                .config("spark.executor.memory", "32g") \
                .config("spark.driver.memory", "32g") \
                .getOrCreate()

    # Load the model
    model_path = "ml_model"
    model = RandomForestRegressionModel.load(model_path)

    # Create a DataFrame with the input values
    input_data = [(vma, acrm)]  # Replace with your desired values
    input_df = spark.createDataFrame(input_data, ["vol_moving_avg", "adj_close_rolling_med"])

    # Select features columns and assemble them into a feature vector
    features = ['vol_moving_avg', 'adj_close_rolling_med']
    assembler = VectorAssembler(inputCols=features, outputCol='features')
    input_df = assembler.transform(input_df).select('features')

    # Make predictions on the input data
    predictions = model.transform(input_df)

    # Display the predictions
    predictions.show()

# Prediction implementation with mlflow
def mlflow_predict(vma, acrm):
    # Path to the MLflow model's artifact location
    model_path = "mlflow_model/"

    # Load the model
    model = mlflow.pyfunc.load_model(model_path)

    # Create a Pandas DataFrame with the input data
    input_data = pd.DataFrame({
        "vol_moving_avg": [vma],
        "adj_close_rolling_med": [acrm]
    })

    # Convert the input data to the appropriate format
    input_data["features"] = input_data.apply(lambda row: Vectors.dense(row["vol_moving_avg"], row["adj_close_rolling_med"]), axis=1)

    # Select only the 'features' column
    input_data = input_data[["features"]]

    # Make predictions using the loaded model
    predictions = model.predict(input_data)

    return predictions

# Prediction implementation with sklearn
def sklearn_predict(vma, acrm):
    filename = 'ml_model.pkl'

    loaded_model = pickle.load(open(filename, 'rb'))

    # Create a DataFrame with the inputs for prediction
    input_data = pd.DataFrame({'vol_moving_avg': [vma], 'adj_close_rolling_med': [acrm]})

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_data)

    return prediction