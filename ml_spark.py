'''
ML implementation using spark.
spark using RDD to train models, which can greatly increase the speed at which a model is trained.

spark ML builds a spark model, which generally requires a Spark sesion to be active in order to serve predictions.
because of this, to serve a spark model through an API you would need to have spark cluster(s) integrated with your backend, which may not
be ideal for most use cases.
'''

from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import logging
from pyspark.ml.feature import VectorAssembler
import mlflow
import logging

def train_model_spark(ml_data):
    try:
        # Set up logging
        logging.basicConfig(filename='training-spark.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s', force=True)

        # Create a Spark session
        spark = SparkSession.builder \
                .master('local') \
                .appName('StockVolumePrediction') \
                .config("spark.executor.memory", "32g") \
                .config("spark.driver.memory", "32g") \
                .getOrCreate()
        
        # Read data into Spark DataFrame
        data = spark.read.parquet(ml_data)
        data = data.withColumn('Date', data['Date'].cast('timestamp'))

        # Remove rows with NaN values
        data = data.dropna()

        # Select features and target
        features = ['vol_moving_avg', 'adj_close_rolling_med']
        target = 'Volume'

        # Select features and target columns and assemble them into a feature vector
        assembler = VectorAssembler(inputCols=features, outputCol='features')
        data = assembler.transform(data).select('features', target)

        # Split the data into train and test sets
        train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

        # Create a RandomForestRegressor model
        model = RandomForestRegressor(labelCol=target, featuresCol='features', numTrees=100, seed=42)

        # Train the model
        trained_model = model.fit(train_data)

        # Make predictions on the test data
        predictions = trained_model.transform(test_data)

        # Evaluate the model's performance
        evaluator = RegressionEvaluator(labelCol=target)
        mae = evaluator.evaluate(predictions, {evaluator.metricName: 'mae'})
        mse = evaluator.evaluate(predictions, {evaluator.metricName: 'mse'})

        # Log training metrics
        logging.info(f'Mean Absolute Error: {mae}')
        logging.info(f'Mean Squared Error: {mse}')

        # Save the trained model
        trained_model.save("ml_model")
        
        # MFLOW
        mlflow.spark.save_model(spark_model=trained_model, path="mlflow_model")
        
        logging.info('Model saved successfully.')

    except Exception as e:
        logging.error(f'An error occurred: {str(e)}')
    finally:
        # Stop the Spark session
        spark.stop()