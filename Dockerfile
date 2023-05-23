FROM prefecthq/prefect:2.7.7-python3.9

COPY docker-requirements.txt .

RUN pip install -r docker-requirements.txt --trusted-host pypi.python.org --no-cache-dir

RUN mkdir -p /root/.kaggle/
RUN mkdir -p /opt/prefect/data/
RUN mkdir -p /opt/prefect/flows/

COPY kaggle_build.py /opt/prefect/flows/kaggle_build.py
# COPY rolling_avg_calculations.py /opt/prefect/flows/rolling_avg_calculations.py
COPY ml_sklearn.py /opt/prefect/flows/ml_sklearn.py
COPY etl_flow.py /opt/prefect/flows/etl_flow.py
# COPY data /opt/prefect/data