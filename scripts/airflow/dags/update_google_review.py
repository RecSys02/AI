import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.models import Variable
import requests

# [중요] 테라폼에서 정의한 값과 일치시킵니다.
BUCKET_NAME = 'ai-park-embeddings-data'
OBJECT_NAME = 'embedding_tourspot.json' 

default_args = {
    'owner': 'developer',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def fetch_google_place_stats(name, lat, lng, api_key):
    """Google Places API (New) 호출"""
    url = "https://places.googleapis.com/v1/places:searchText"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "places.id,places.rating,places.userRatingCount"
    }
    payload = {
        "textQuery": name,
        "locationBias": {
            "circle": {
                "center": {"latitude": lat, "longitude": lng},
                "radius": 500.0
            }
        }
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        places = response.json().get("places", [])
        return places[0] if places else None
    except Exception as e:
        logging.error(f"API Error for {name}: {e}")
        return None

def update_embeddings_on_gcs(**context):
    # GCP 연결 훅 생성
    gcs_hook = GCSHook(gcp_conn_id='google_cloud_default')
    api_key = Variable.get("google_api_key")

    # 1. GCS에서 현재 데이터 다운로드 (메서드 수정 완료)
    try:
        # download_as_bytearray 대신 download를 사용합니다.
        file_content = gcs_hook.download(bucket_name=BUCKET_NAME, object_name=OBJECT_NAME)
        data = json.loads(file_content.decode('utf-8'))
    except Exception as e:
        logging.error(f"GCS Download Error: {e}")
        # 에러 발생 시 태스크를 실패 처리하기 위해 raise를 던지는 것을 추천합니다.
        raise e

    updated_count = 0
    limit = 100 

    for item in data:
        if updated_count >= limit:
            break
            
        google_info = item.get('google', {})
        if not google_info or not google_info.get('place_id'):
            name = item.get('name')
            loc = item.get('location', {})
            
            if name and loc.get('lat') and loc.get('lng'):
                res = fetch_google_place_stats(name, loc['lat'], loc['lng'], api_key)
                
                if res:
                    item['google'] = {
                        "place_id": res.get("id"),
                        "rating": res.get("rating"),
                        "user_ratings_total": res.get("userRatingCount"),
                        "last_updated": datetime.now().strftime("%Y-%m-%d")
                    }
                    updated_count += 1
                    logging.info(f"Updated: {name}")

    # 2. 업데이트된 데이터를 다시 GCS에 업로드
    if updated_count > 0:
        updated_json = json.dumps(data, ensure_ascii=False, indent=2)
        gcs_hook.upload(
            bucket_name=BUCKET_NAME,
            object_name=OBJECT_NAME,
            data=updated_json,
            content_type='application/json'
        )
        logging.info(f"Successfully updated {updated_count} items in GCS.")
    else:
        logging.info("No items were updated.")

with DAG(
    dag_id="sync_gcs_embeddings_with_google_api",
    default_args=default_args,
    schedule_interval="@weekly",
    catchup=False,
    tags=['gcs', 'cloud_run', 'places_api']
) as dag:

    update_task = PythonOperator(
        task_id="update_places_data",
        python_callable=update_embeddings_on_gcs
    )