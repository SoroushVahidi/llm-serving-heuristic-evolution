# GPU Environment

Generated: 2026-06-10T20:15:28.210547+00:00

## Hardware

- GPU 0: **NVIDIA GeForce RTX 5060 Ti** — 15.48 GB VRAM, 36 SMs, compute 12.0

- Driver version: 580.142
- CUDA version: 13.0
- RAM total: 62.5 GB, free: 56.8 GB
- Disk free: 594.7 GB

## Software

- Python: 3.12.3
- PyTorch: 2.12.0+cu130
- Transformers: 5.8.1
- Accelerate: 1.13.0
- vLLM: not installed

## HuggingFace Cache

- Cache dir: `/home/soroush/.cache/huggingface`
- Cached models (8):
  - .locks
  - datasets--HuggingFaceH4--MATH-500
  - datasets--HuggingFaceH4--aime_2024
  - datasets--Idavidrein--gpqa
  - datasets--TIGER-Lab--MMLU-Pro
  - datasets--openai--gsm8k
  - models--sentence-transformers--all-MiniLM-L6-v2
  - models--sentence-transformers--all-mpnet-base-v2

## nvidia-smi

```
Wed Jun 10 16:15:26 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.142                Driver Version: 580.142        CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti     Off |   00000000:01:00.0 Off |                  N/A |
|  0%   30C    P8              3W /  180W |      15MiB /  16311MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1868      G   /usr/lib/xorg/Xorg                        4MiB |
+-----------------------------------------------------------------------------------------+
```
