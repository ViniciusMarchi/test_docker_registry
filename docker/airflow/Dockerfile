FROM apache/airflow:2.10.3

COPY dags /opt/airflow/dags

RUN pip install apache-airflow[kubernetes] kubernetes

USER airflow