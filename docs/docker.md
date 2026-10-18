# Docker Usage Guide

**Cattle Vision Framework** — MS Thesis, Sakif Khan, Texas State University 2026

The pipeline is containerized with one Docker image per stage group. All images share a
common base (`pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`) and mount `data/`, `runs/`,
and `results/` from the host.

---

## 1. Prerequisites

- Docker ≥ 24
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU stages
- At least 20 GB free disk for image layers

**Verify GPU access:**

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Fix Docker context (Linux + Docker Desktop):**

```bash
# Docker Desktop uses a different daemon than the system dockerd.
# GPU stages require the system dockerd with the nvidia runtime.
docker context use default
docker info | grep -i runtime   # must show: nvidia
```

If `nvidia` is missing from the system dockerd:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 2. Images

| Image name            | Dockerfile                       | Pipeline stages           | Key deps                            |
| --------------------- | -------------------------------- | ------------------------- | ----------------------------------- |
| `cattle-data`         | `docker/Dockerfile.data`         | 1–4, 9, 12                | opencv, pycocotools, pandas, pyyaml |
| `cattle-detection`    | `docker/Dockerfile.detection`    | 5–6                       | rfdetr, supervision, albumentations |
| `cattle-segmentation` | `docker/Dockerfile.segmentation` | 7a (SAM2)                 | sam2, opencv                        |
| `cattle-rfdetr-seg`   | `docker/Dockerfile.rfdetr_seg`   | 7b (RF-DETR-Seg training) | rfdetr                              |
| `cattle-tracking`     | `docker/Dockerfile.tracking`     | 8 (OC-SORT)               | lap, filterpy                       |
| `cattle-behavior`     | `docker/Dockerfile.behavior`     | 10–11 (VideoMAE)          | transformers≥4.40, timm             |

---

## 3. Build Images

Build from the repo root. All Dockerfiles use `..` as context so they can access `src/`, `configs/`, etc.

```bash
# Build all images
docker compose -f docker/docker-compose.yml build

# Build a single image
docker compose -f docker/docker-compose.yml build detection
docker compose -f docker/docker-compose.yml build behavior
```

Or build directly with Docker:

```bash
docker build -f docker/Dockerfile.detection -t cattle-detection .
docker build -f docker/Dockerfile.behavior  -t cattle-behavior  .
```

---

## 4. Run a Single Pipeline Stage

All stages are run via `docker compose run`. The working directory inside each container is `/workspace`.

```bash
# Stage 1 — inspect data
docker compose -f docker/docker-compose.yml run --rm data bash scripts/01_inspect_data.sh

# Stage 2 — prepare CBVD-5
docker compose -f docker/docker-compose.yml run --rm data bash scripts/02_prepare_cbvd5.sh

# Stage 5 — train detector (GPU required)
docker compose -f docker/docker-compose.yml run --rm detection bash scripts/05_train_detector.sh

# Stage 7a — SAM2 segmentation (GPU required)
docker compose -f docker/docker-compose.yml run --rm segmentation bash scripts/07_run_segmentation.sh

# Stage 9 — generate tubelets (long-running, CPU)
docker compose -f docker/docker-compose.yml run --rm data bash scripts/09_generate_tubelets.sh

# Stage 10 — train VideoMAE (GPU required, long-running)
docker compose -f docker/docker-compose.yml run --rm behavior bash scripts/10_train_behavior.sh

# Stage 11 — evaluate behavior classifiers
docker compose -f docker/docker-compose.yml run --rm behavior bash scripts/11_evaluate.sh

# Stage 12 — generate analytics
docker compose -f docker/docker-compose.yml run --rm data bash scripts/12_generate_analytics.sh
```

---

## 5. Run the Full Pipeline

```bash
bash scripts/run_pipeline.sh             # run all 12 stages
bash scripts/run_pipeline.sh --from 9   # resume from stage 9
bash scripts/run_pipeline.sh --stage 11 # run just stage 11
```

`run_pipeline.sh` calls `docker compose run` for each stage in order.

---

## 6. Volume Mounts

All services mount the following host directories into `/workspace/` inside the container:

| Host path   | Container path       | Mode                         |
| ----------- | -------------------- | ---------------------------- |
| `./data`    | `/workspace/data`    | read-write                   |
| `./results` | `/workspace/results` | read-write                   |
| `./runs`    | `/workspace/runs`    | read-write (GPU stages only) |
| `./configs` | `/workspace/configs` | read-only                    |
| `./src`     | `/workspace/src`     | read-only                    |
| `./models`  | `/workspace/models`  | read-only                    |

The `tracking` service additionally mounts `third_party/OC_SORT/`:

| Host path               | Container path                   | Mode      |
| ----------------------- | -------------------------------- | --------- |
| `./third_party/OC_SORT` | `/workspace/third_party/OC_SORT` | read-only |

---

## 7. GPU Configuration

GPU stages use `runtime: nvidia` in `docker-compose.yml`. By default they use all available GPUs.
To restrict to a specific GPU:

```bash
# Use only GPU 0
NVIDIA_VISIBLE_DEVICES=0 docker compose -f docker/docker-compose.yml run --rm behavior \
  bash scripts/10_train_behavior.sh

# Or pass directly
docker run --rm --gpus '"device=0"' ... cattle-behavior ...
```

---

## 8. Shared Memory (--shm-size)

VideoMAE training uses multiple DataLoader workers and requires at least 16 GB of shared
memory. The `behavior` service in `docker-compose.yml` sets `shm_size: 16g`.

If running `docker run` directly, always include `--shm-size=16g`:

```bash
docker run --rm --gpus all --shm-size=16g \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/runs:/workspace/runs \
  -v $(pwd)/configs:/workspace/configs:ro \
  -v $(pwd)/src:/workspace/src:ro \
  cattle-behavior \
  python src/behavior/train.py --config configs/behavior/videomae_combined.yaml
```

Forgetting `--shm-size` causes `RuntimeError: No space left on device` in the DataLoader.

---

## 9. Troubleshooting

### "could not select device driver" on local machine

Docker Desktop is the active context. Switch to system Docker:

```bash
docker context use default
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### "libnvidia-ml.so.1: cannot open shared object file"

ldconfig cache is stale on the server:

```bash
sudo ldconfig
find /usr /lib /lib64 -name 'libnvidia-ml.so*' 2>/dev/null
# Add the found directory to /etc/ld.so.conf.d/ then:
sudo ldconfig
```

### CUDA out of memory (OOM) on RTX 3060 (12 GB)

- Reduce batch size: set `batch_size: 2` in config YAML or pass as env var
- Reduce image resolution to 512 (must be divisible by 64)
- BF16 mixed precision and gradient checkpointing must be enabled (already default in all behavior configs)

### Build fails on `pip install rfdetr`

`rfdetr` requires PyTorch to be installed first. The Dockerfiles install torch before rfdetr —
if building manually, ensure `torch==2.8.0` is installed before `rfdetr==1.4.3`.

### OC-SORT not found inside tracking container

The tracking service mounts `third_party/OC_SORT/` at build time. Ensure you've cloned it:

```bash
git clone https://github.com/noahcao/OC_SORT.git third_party/OC_SORT
```

Then rebuild or re-run the service.
