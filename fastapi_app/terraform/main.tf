resource "google_project_service" "artifact_registry" {
  service = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloud_run" {
  service = "run.googleapis.com"
  disable_on_destroy = false
}

terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = "gen-lang-client-0492042254"
  region  = "asia-northeast3"
}

# 1. Docker 이미지 저장소
resource "google_artifact_registry_repository" "ai_repo" {
  location      = "asia-northeast3"
  repository_id = "ai-server"
  description   = "Docker repository for AI POI Recommendation API"
  format        = "DOCKER"
}

# 2. 임베딩 데이터를 저장할 GCS 버킷 생성
resource "google_storage_bucket" "data_bucket" {
  name          = "ai-park-embeddings-data" 
  location      = "ASIA-NORTHEAST3"
  force_destroy = true 
}

# 3. Cloud Run 서비스 정의
resource "google_cloud_run_v2_service" "ai_service" {
  name     = "ai-server-service"
  location = "asia-northeast3"
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    containers {
      image = "asia-northeast3-docker.pkg.dev/gen-lang-client-0492042254/ai-server/app:latest"
      
      ports {
        container_port = 8000
      }

      env {
        name  = "OPENAI_API_KEY"
        value = "your-api-key-here" # 나중에 Secret Manager로 교체 권장
      }

      volume_mounts {
        name       = "embeddings-storage"
        mount_path = "/app/data" 
      }
    }

    # volumes 블록은 containers 블록과 같은 레벨(template 바로 아래)에 있어야 합니다.
    volumes {
      name = "embeddings-storage"
      gcs {
        bucket    = google_storage_bucket.data_bucket.name
        read_only = false # 필요에 따라 설정
      }
    }
  }
}

output "repository_url" {
  value = "${google_artifact_registry_repository.ai_repo.location}-docker.pkg.dev/gen-lang-client-0492042254/${google_artifact_registry_repository.ai_repo.repository_id}"
}

output "bucket_name" {
  value = google_storage_bucket.data_bucket.name
}