# Stock Volume Machine Learning ETL Pipeline

## Table of contents
1. [Purpose of the Project](#purpose)
2. [Data Pipeline Architecture](#architecture)
3. [Raw Data Processing](#raw)
4. [Feature Engineering](#feat)
5. [Integrating Machine Learning](#ml)
6. [Steps to Reproduce](#repro)
7. [Serving the Model](#serve)
8. [Prefect](#prefect)

## Purpose of the Project <a name='purpose'></a>
The idea behind this project is to build a data pipeline that extracts historical stock data from a kaggle dataset https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset for the purpose of training a machine learning model to predict stock volume at any given time, given two input parameters (vol_moving_avg and adj_close_rolling_med).

## Data Pipeline Architecture <a name='architecture'></a>
The pipeline architecture can be seen in the below diagram. Technologies that will be used include:
* <b>Prefect</b> - For Flow and Task Management (Optional)
* <b>Docker</b> - Containerization
* <b>Spark</b> - For building the ML model and RDD (Choice)
* <b>Sci-kit Learn</b> - For building the ML model (Choice)
* <b>Flask</b> - For serving the ML model

![image](https://github.com/Phillip-N/de-work-sample/assets/10274304/0bbd1c3b-c574-488e-9848-638a7ac9ca7a)

## Raw Data Processing <a name='raw'></a>
Raw data will be downloaded from Kaggle using the Kaggle API. In order for this to work, the user will need to open up a kaggle account and generate an API token https://www.kaggle.com/docs/api. The credentials can be later passed down as a parameter to the docker container.

At this stage, the dataset is simply being combined with the following columns being retained. The docker image will save down the combined dataset as a parquet file in /usr/app/src/data/, and name it solution-1.parquet. To copy the combined dataset locally you can use `docker cp {CONTAINER_NAME}:/usr/app/src/data/solution-1.parquet {LOCAL_PATH}`
```
Symbol: string
Security Name: string
Date: string (YYYY-MM-DD)
Open: float
High: float
Low: float
Close: float
Adj Close: float
Volume: A suitable Number type (int or float)
```

## Feature Engineering <a name='feat'></a>
The combined dataset will then be used to calculate the value of two additional metrics `vol_moving_avg` and `adj_close_rolling_med`. This will leverage pandas' built in rolling method, to calculate both of these moving averages within a 30-day window.

Before calculating these metrics, the dataset is cleaned to remove rows where there is a null value in the Open, High, Low, Close, Adj Close and Volume columns. The reason being that null values in these columns do not make sense in the case of stock market data, and some of these data points would be required to effectively calculate `vol_moving_avg` and `adj_close_rolling_med`. We will ignore null values in Security Name as it is not neccessary for our calculation.

Due to the size of the dataset, and the fact that we are using the combined dataset as the starting point, we can expect the rolling function to perform very slowly on your typical system. To speed up the process, a few enhacements were made.
1. Multiprocessing was implemented, so that jobs can be worked on in parallel.
2. File was split up into smaller files (grouped by ticker symbol) so that workers can retrieve data indepedently.
3. `pool.close()` and `pool.join()` were used to avoid memory leaks within the docker container, as sometimes the workers refused to release resources. This makes sure enough memory is available for training the ML model later on

After the calculations have been done and the new columns have been added, the new dataset is saved down as solution-2.parquet in /usr/app/src/data/. You can use `docker cp {CONTAINER_NAME}:/usr/app/src/data/solution-2.parquet {LOCAL_PATH}` to save it locally.


## Integrating Machine Learning <a name='ml'></a>
Using a Random Forest algorithm, we can train a model to predict stock volume based the `vol_moving_avg` and `adj_close_rolling_med` metrics we calculated above. Two options were considered when deciding which machine learning library to use to build the model.

**Spark**
pyspark can be used to train the model, and leverages Spark's resilient distributed dataset for parallel computer. The benefit of using Spark from what I've seen is that it able to more efficiently utilize resources when dealing with larger datasets. The time it takes for Spark to build 100 trees with no limit on depth, was much faster than Sci-kit learn. A Spark variation of the ML training script is available under `ml_spark.py`, however this has not been integrated with docker.

**Sci-kit Learn**
Sci-kit learn can also be used to train the model. The benefit of sci-kit learn is that it requires much less overhead than Spark does, and for use-cases where the dataset is small, and where training periods are not a huge concern, it may be preferable. Using Sklearn also means that the model can easily be seralized (pickled) and deserialized once it has finished training. This makes it much more easier to serve the model on an API for example, because unlike with Spark models, sklearn models do not require a Spark session and clusters to be active.

Because we plan to serve the trained model through an API, we will opt for using sci-kit learn to train the model, and pickle it for later use. The following parameters were used to train the model:
* <b>n_estimators=100</b>
* <b>random_state=42</b>
* <b>n_jobs=-1</b> - to ensure the maximum amount of cores is used for parallel processing
* <b>max_depth=6</b> - setting max_depth to 6 as to not overfit the model, and also reduce training time
* <b>verbose=10</b> - for printing out progress to console

### Steps to Reproduce <a name='repro'></a>
As the pipeline has been entirely dockerized, reproducing the results should be a matter of following the below steps.

1. Pull the image from docker hub here: https://hub.docker.com/repository/docker/phillipng/stock-etl-docker/general
2. Run the docker image, passing in your kaggle username and key (found in .json file, these should be strings and passed as "username" or "password") `docker run --shm-size=8g -e KAGGLE_USER={YOUR_KAGGLE_USERNAME} -e KAGGLE_KEY={YOUR_KAGGLE_KEY} phillipng/stock-etl-docker:{tag}`
3. This will run the etl pipeline and print out progress updates to the console (progress also trackable through the Pefect UI, if using Prefect)
4. Any files of interest can then be copied locally
   * `docker cp {DOCKER_CONTAINER_NAME}:/usr/app/src/data/solution-1.parquet {LOCAL_PATH}`
   * `docker cp {DOCKER_CONTAINER_NAME}:/usr/app/src/data/solution-2.parquet {LOCAL_PATH}`
   * `docker cp {DOCKER_CONTAINER_NAME}:/usr/app/src/ml_model.pkl {LOCAL_PATH}`
   * `docker cp {DOCKER_CONTAINER_NAME}:/usr/app/src/training-sklearn.log {LOCAL_PATH}`

### Serving the Machine Learning Model <a name='serve'></a>
The model was served using a basic flask app on render.com. Currently, the web app has a single endpoint, which **requires** two input parameters `vol_moving_avg` and `adj_close_rolling_med`, that can be either an integer or a float. The endpoint uri must look like the below for the GET call to be successful. 

https://stock-volume-predict.onrender.com/predict?vol_moving_avg=4213434&adj_close_rolling_med=532

### BONUS: Using Prefect <a name='prefect'></a>
Prefect, which can be thought of as an alternative to airflow, can be used to more easily track the workflow of our ETL pipeline, while also allowing us to better manage our infrastructure with other services. Both a prefect variant of the `etl_flow.py` (prefect version: `etl_flow_prefect.py`) script and the dockerfile (`Dockerfile_prefect`), can also be found in this repo, and can be used if the user prefers a more cleaner approach to managing their workflows.

Using prefect requires the user know how to set properly set up their docker blocks https://docs.prefect.io/latest/guides/deployment/docker/ and requires a prefect orion server and a prefect agent to be active. Once these prerequisites are met, we can sit back while Prefect manages the workflow.

![flow_run](https://github.com/Phillip-N/de-work-sample/assets/10274304/957ba4b5-2248-4312-bd92-cf426eb67c81)




