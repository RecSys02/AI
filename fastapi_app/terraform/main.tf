# Compute Engine API 활성화 리소스 추가
resource "google_project_service" "compute_engine" {
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

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

# Airflow 서버를 위한 VM 인스턴스
resource "google_compute_instance" "airflow_vm" {
  name         = "airflow-server"
  machine_type = "e2-medium" # Airflow 실행을 위한 최소 권장 사양
  zone         = "asia-northeast3-a"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 30 # 디스크 용량 (GB)
    }
  }

  network_interface {
    network = "default"
    access_config {
      # 외부 IP를 할당하여 SSH 접속이 가능하게 함
    }
  }

  # VM에 GCS 및 API 호출 권한 부여 (이미 만들어진 서비스 계정이 있다면 해당 이메일 사용)
  service_account {
    scopes = ["cloud-platform"]
  }

  # [중요] VM 시작 시 Docker 및 Docker Compose 자동 설치 스크립트
  metadata_startup_script = <<-EOF
    #!/bin/bash
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/local/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo usermod -aG docker $USER
  EOF

  # 외부에서 Airflow 웹 UI(8080 포트)에 접속할 수 있도록 태그 추가
  tags = ["airflow-web"]

  depends_on = [google_project_service.compute_engine]
}

# 방화벽 설정 (8080 포트 개방)
resource "google_compute_firewall" "airflow_firewall" {
  name    = "allow-airflow-web"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["8080"]
  }

  source_ranges = ["0.0.0.0/0"] # 실제 운영 시에는 본인 IP만 허용하는 것이 안전합니다.
  target_tags   = ["airflow-web"]
  depends_on = [google_project_service.compute_engine]
}