resource "google_project_service" "artifact_registry" {
  service            = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloud_run" {
  service            = "run.googleapis.com"
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
    timeout = "600s"

    containers {
      image = "asia-northeast3-docker.pkg.dev/gen-lang-client-0492042254/ai-server/app:latest"

      resources {
        limits = {
          memory = "8Gi"
          cpu    = "4"
        }
      }

      ports {
        container_port = 8000
      }

      volume_mounts {
        name       = "embeddings-storage"
        mount_path = "/data"
      }
    }

    volumes {
      name = "embeddings-storage"
      gcs {
        bucket    = google_storage_bucket.data_bucket.name
        read_only = false
      }
    }
  } # template 블록 끝

  # GitHub Actions에서 주입하는 환경변수와 라벨이 삭제되지 않도록 보호
  lifecycle {
    ignore_changes = [
      template[0].containers[0].env,
      template[0].labels,
      template[0].annotations,
    ]
  }
}

output "repository_url" {
  value = "${google_artifact_registry_repository.ai_repo.location}-docker.pkg.dev/gen-lang-client-0492042254/${google_artifact_registry_repository.ai_repo.repository_id}"
}

output "bucket_name" {
  value = google_storage_bucket.data_bucket.name
}

resource "google_cloud_run_v2_service_iam_member" "public_access" {
  location = google_cloud_run_v2_service.ai_service.location
  name     = google_cloud_run_v2_service.ai_service.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}