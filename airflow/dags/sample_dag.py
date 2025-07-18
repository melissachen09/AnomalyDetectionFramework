"""
Sample DAG file for Anomaly Detection Framework

This is a placeholder DAG to ensure the DAGs directory is properly configured.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.dummy import DummyOperator

default_args = {
    'owner': 'anomaly-detection-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'sample_anomaly_detection',
    default_args=default_args,
    description='Sample DAG for Anomaly Detection Framework',
    schedule_interval=timedelta(days=1),
    catchup=False
)

start_task = DummyOperator(
    task_id='start',
    dag=dag
)

end_task = DummyOperator(
    task_id='end',
    dag=dag
)

start_task >> end_task