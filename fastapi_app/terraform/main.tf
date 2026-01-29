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
# Airflow 서버를 위한 VM 인스턴스
resource "google_compute_instance" "airflow_vm" {
  name         = "airflow-server"
  machine_type = "e2-medium"
  zone         = "asia-northeast3-a"

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 30
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  service_account {
    scopes = ["cloud-platform"]
  }
    metadata = {
        "ssh-keys" = "roto9379:${file("./id_rsa_gcp.pub")}"
    }
  # [개선된] Docker 공식 저장소 등록 및 최신 패키지 설치 스크립트
  metadata_startup_script = <<-EOF
    #!/bin/bash
    set -e  # 에러 발생 시 즉시 중단

    # 1. 필수 패키지 설치 및 GPG 키 등록 준비
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg

    # 2. Docker 공식 GPG 키 추가
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg --yes
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    # 3. Docker 저장소 추가
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # 4. 최신 Docker 패키지 설치
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # 5. 사용자 그룹 권한 부여 (VM 접속 계정명 확인 필요, 여기서는 roto9379를 예시로 추가)
    sudo groupadd docker || true
    sudo usermod -aG docker roto9379
  EOF

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