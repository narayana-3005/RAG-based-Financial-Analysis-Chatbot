from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os


# Import the custom modules for fetching data

dag_path = '/Users/karanbadlani/airflow'
sys.path.insert(0, f"{dag_path}/scripts")

# Set environment variables for the API key and Google Cloud credentials
os.environ['POLYGON_API_KEY'] = 'kbU9wA0bTPzIwQxCyrIacqu4yYkuriwl'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "google_creds.json"

from fetch_news_articles import fetch_and_store_news
from fetch_stock_data import store_data_to_gcs
from stockprices_preprocessing import preprocess_all_data
from NewArticles2 import preprocess_all_news_data

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'data_pipeline',
    default_args=default_args,
    description='A data pipeline for acquiring stock and news data, storing in GCP buckets and Preprocessing the data files',
    schedule_interval=timedelta(days=1),  # Adjust schedule as needed
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    # # Task to fetch stock data
    # def fetch_stock_data_task():
    #     fetch_stock_data.store_data_to_gcs()  

    # #Task to fetch news articles
    # def fetch_news_articles_task():
    #     fetch_news_articles.fetch_and_store_news()  

    # Define the tasks for the DAG
    fetch_stock_data = PythonOperator(
        task_id='fetch_stock_data',
        python_callable=store_data_to_gcs
    )

    fetch_news_articles = PythonOperator(
        task_id='fetch_news_articles',
        python_callable=fetch_and_store_news
    )

    fetch_stock_data_preprocessing = PythonOperator(
        task_id='fetch_stock_data_preprocessing',
        python_callable=preprocess_all_data
    )

    fetch_news_data_preprocessing = PythonOperator(
        task_id='fetch_news_data_preprocessing',
        python_callable=preprocess_all_news_data 
    )

    # Set task dependencies
    fetch_stock_data >> fetch_news_articles >> fetch_stock_data_preprocessing >> fetch_news_data_preprocessing # This runs fetch_stock_data first, then fetch_news_articles
    