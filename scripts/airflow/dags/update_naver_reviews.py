from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="update_naver_reviews",
    start_date=datetime(2024, 1, 1),
    schedule="0 3 * * *",
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=10)},
) as dag:
    BashOperator(
        task_id="update_tourspot_reviews",
        bash_command=(
            "python /opt/airflow/repo/scripts/airflow/update_naver_reviews.py "
            "--playwright --sleep 0.2"
        ),
    )
