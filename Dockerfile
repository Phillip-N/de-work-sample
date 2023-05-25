FROM python:3.10.1

COPY docker-requirements.txt .

RUN pip install -r docker-requirements.txt --trusted-host pypi.python.org --no-cache-dir

RUN mkdir -p /root/.kaggle/
RUN mkdir -p /usr/app/src/data/
RUN mkdir -p /usr/app/src/data/job_files

COPY kaggle_build.py /usr/app/src/kaggle_build.py
COPY ml_sklearn.py /usr/app/src/ml_sklearn.py
COPY etl_flow.py /usr/app/src/etl_flow.py

WORKDIR /usr/app/src
CMD [ "python", "-u", "./etl_flow.py"]
