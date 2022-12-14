from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import pendulum

with DAG(
    dag_id='batch_pipeline',
    default_args={'retries':2},
    description='sensor fault detection',
    schedule_interval='@weekly',
    start_date=pendulum.datetime(2022, 12, 14, tz='UTC'),
    catchup=False,
    tags=['example'],
) as dag:

    def download_files(**kwargs):
        bucket_name = os.getenv('BUCKET_NAME')
        input_dir = '/app/input_files'
        os.makedirs(input_dir, exist_ok=True)
        os.system(f'aws s3 sync s3://{bucket_name}/input_files /app/input_files')

    def batch_prediction(**kwargs):
        from sensor.pipeline.batch_prediction import start_batch_prediction
        input_dir = '/app/input_files'
        for file_name in os.listdir(input_dir):
            start_batch_prediction(input_file_path=os.path.join(input_dir, file_name))


    def sync_with_s3_bucket(**kwargs):
        bucket_name = os.getenv('BUCKET_NAME')
        os.system(f'aws s3 sync /app/prediction s3://{bucket_name}/prediction_files')

    download_file = PythonOperator(
        task_id='download_file',
        python_callable=download_files
    )

    generate_batch_prediction = PythonOperator(
        task_id='prediction',
        python_callable=batch_prediction
    )

    upload_prediction_files = PythonOperator(
        task_id='upload_predictions',
        python_callable=sync_with_s3_bucket
    )

    download_file >> generate_batch_prediction >> upload_prediction_files
