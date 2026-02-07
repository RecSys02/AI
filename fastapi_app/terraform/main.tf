# 1. 프로젝트 및 Provider 설정
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

# 2. 필수 API 활성화
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

resource "google_project_service" "sqladmin" {
  service            = "sqladmin.googleapis.com"
  disable_on_destroy = false
}

# 3. 변수 설정
variable "db_user" {
  type    = string
  default = "poi_user"
}

variable "db_password" {
  type      = string
  sensitive = true
}

variable "db_name" {
  type    = string
  default = "poi_meta"
}

# 4. Artifact Registry
resource "google_artifact_registry_repository" "ai_repo" {
  location      = "asia-northeast3"
  repository_id = "ai-server"
  description   = "Docker repository for AI POI Recommendation API"
  format        = "DOCKER"
}

# 5. GCS 버킷
resource "google_storage_bucket" "data_bucket" {
  name          = "ai-park-embeddings-data"
  location      = "ASIA-NORTHEAST3"
  force_destroy = true
}

# 6. Cloud SQL (Postgres)
resource "google_sql_database_instance" "poi_postgres" {
  name             = "poi-postgres"
  database_version = "POSTGRES_15"
  region           = "asia-northeast3"
  deletion_protection = false # 삭제 가능하도록 설정

  settings {
    tier      = "db-custom-1-3840"
    disk_type = "PD_SSD"
    disk_size = 20
    ip_configuration {
      ipv4_enabled = true
    }
  }
  depends_on = [google_project_service.sqladmin]
}

resource "google_sql_database" "poi_db" {
  name     = var.db_name
  instance = google_sql_database_instance.poi_postgres.name
}

resource "google_sql_user" "poi_user" {
  name     = var.db_user
  password = var.db_password
  instance = google_sql_database_instance.poi_postgres.name
}

# 7. Cloud Run 서비스 (비용 최적화 버전)
resource "google_cloud_run_v2_service" "ai_service" {
  name     = "ai-server-service"
  location = "asia-northeast3"
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    timeout = "600s"

    scaling {
      # [변경] 1 -> 0: 요청이 없으면 서버를 완전히 꺼서 0원 청구
      min_instance_count = 0 
      max_instance_count = 2
    }

    containers {
      image = "asia-northeast3-docker.pkg.dev/gen-lang-client-0492042254/ai-server/app:latest"

      resources {
        limits = {
          # [변경] 사양 하향: 4 CPU / 8Gi -> 1 CPU / 2Gi
          memory = "4Gi"
          cpu    = "2"
        }
        # [변경] false -> true: 유휴 상태일 때 CPU 비용 지불 안 함
        cpu_idle = true 
      }

      ports {
        container_port = 8000
      }

      volume_mounts {
        name       = "embeddings-storage"
        mount_path = "/data"
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }
    }

    volumes {
      name = "embeddings-storage"
      gcs {
        bucket    = google_storage_bucket.data_bucket.name
        read_only = false
      }
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.poi_postgres.connection_name]
      }
    }
  }

  lifecycle {
    ignore_changes = [
      template[0].containers[0].env,
      template[0].labels,
      template[0].annotations,
    ]
  }
}

resource "google_cloud_run_v2_service_iam_member" "public_access" {
  location = google_cloud_run_v2_service.ai_service.location
  name     = google_cloud_run_v2_service.ai_service.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# 8. Milvus 서버 VM (사양 하향 조정)
resource "google_compute_instance" "milvus_vm" {
  name         = "milvus-server"
  # [변경] e2-standard-4 -> e2-standard-2 (비용 약 50% 절감)
  machine_type = "e2-standard-2" 
  zone         = "asia-northeast3-a"

  allow_stopping_for_update = true
  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 200 # 용량 축소는 데이터 유실 위험이 있어 유지
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  service_account {
    scopes = ["cloud-platform"]
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg --yes
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  EOF

  tags = ["milvus"]
  depends_on = [google_project_service.compute_engine]
}

# 9. Milvus 방화벽 설정
resource "google_compute_firewall" "milvus_firewall" {
  name    = "allow-milvus"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["19530", "9091"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["milvus"]
  depends_on = [google_project_service.compute_engine]
}

# Outputs
output "cloudsql_instance_connection_name" {
  value = google_sql_database_instance.poi_postgres.connection_name
}