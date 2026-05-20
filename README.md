# MeCo

<p align="left">
  <a href="https://arxiv.org/abs/2503.09027">
    <img src="https://img.shields.io/badge/arXiv-2503.09027-b31b1b.svg" alt="arXiv">
  </a>
</p>

This repository maintains the official implementation of the paper **Measure Twice, Cut Once: A Semantic-Oriented Approach to Video Temporal Localization with Video LLMs (ICLR 2026)**

## 📋 To-Do List
- [x] Upload code
- [x] Upload data  
- [ ] Upload checkpoint

## 🛠️ Installation

Please refer to the following environmental settings that we use. You may install these packages by yourself if you meet any problem during automatic installation.

- CUDA 11.8
- Python 3.10
- PyTorch 2.1.2
- [Transformers](https://github.com/huggingface/transformers) 4.44.2
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) 0.14.5
- [NNCore](https://github.com/yeliudev/nncore) 0.4.5

### Install from source

1. Clone the repository from GitHub.

```shell
git clone https://github.com/pangzss/MeCo.git
cd MeCo
```

2. Initialize conda environment.

```shell
conda create -n meco python=3.10 -y
conda activate meco
```

3. Install dependencies.

```shell
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 🚀 Getting Started

Following [ETChat](https://github.com/PolyU-ChenLab/ETBench/tree/main), we apply a three-stage training recipe for MeCo, where the first stage is for modality alignment, the second stage is for acquiring general chatting abilities, and the third stage is for enabling temporal localization capabilities.

### Prepare model checkpoints

We compare the learnable modules in each stage, and provide their checkpoints as follows.

|| Encoder | Q-Former | Aggregator | Projector | LLM (LoRA) | Checkpoint |
|-|:-:|:-:|:-:|:-:|:-:|:-:|
| `Stage-1` | ❄️ | ❄️ | 🔥 | 🔥 | ❄️ | [4B](TBD) / [7B](TBD) |
| `Stage-2` | ❄️ | 🔥 | 🔥 | 🔥 | 🔥 | [4B](TBD) / [7B](TBD) |
| `Stage-3` | ❄️ | 🔥 / ❄️ | 🔥 | 🔥 | 🔥 | [4B](TBD) / [7B](TBD) |

If you want to start from `stage-1`, the pre-trained weights from [Phi3-Mini-4K-Instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct), [MiniCPM-V-2_6_QWen2](TBD), [EVA-ViT-G](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth), and [Q-Former](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth) are required for initializing the model. The downloaded checkpoints shall be saved in the `checkpoints` folder.

### Training datasets

The training data used in each stage follows a similar structure to E.T. Chat. We use the ET-Instruct-164K dataset for stage-3 training with additional strutural token and query-focused captioning annotations.

|| Video Data | Image Data | Annotations |
|-|-|-|-|
| `Stage-1` | [WebVid](https://maxbain.com/webvid-dataset/) | [LCS-558K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) | [llava_558k_with_webvid.json](https://huggingface.co/datasets/YanweiLi/LLaMA-VID-Data/resolve/main/llava_558k_with_webvid.json) |
| `Stage-2` | [ActivityNet](http://activity-net.org/download.html) / [VideoChatGPT](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EnLRDehrr8lGqHpC5w1zZ9QBnsiVffYy5vCv8Hl14deRcg?e=Ul5DUE) | [LLaVA-1.5-Instruct](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#visual-instruction-tuning) | [llava_v1_5_mix665k_with_video_chatgpt.json](https://huggingface.co/datasets/YanweiLi/LLaMA-VID-Data/resolve/main/llava_v1_5_mix665k_with_video_chatgpt.json) |
| `Stage-3` | [ET-Instruct-164K](https://huggingface.co/datasets/PolyU-ChenLab/ET-Instruct-164K) | - | [et_instruct_164k_meco.json](https://huggingface.co/datasets/pangzs/MeCo) |

Download the required datasets and place them in the `data` folder. It is strongly recommended to compress the videos (to `3 FPS` & `224px`) using the [script](tools/compress_videos.py) provided. After processing, make sure the files are organized in the following structure.

```
MeCo
├─ data
│  ├─ llamavid
│  │  ├─ llava_558k_with_webvid.json
│  │  └─ llava_v1_5_mix665k_with_video_chatgpt.json
│  ├─ llava_pretrain                 ─┐
│  │  └─ images                       │ For
│  ├─ webvid                          │ Stage-1
│  │  └─ videos                      ─┘
│  ├─ llava_instruct                 ─┐
│  │  ├─ coco                         │
│  │  ├─ gqa                          │
│  │  ├─ ocr_vqa                      │ For
│  │  ├─ textvqa                      │ Stage-2
│  │  └─ vg                           │
│  ├─ video_chatgpt                   │
│  │  └─ activitynet                 ─┘
│  ├─ ET-Instruct-164K               ─┐
│  │  ├─ videos                       │
│  │  ├─ et_instruct_164k_txt.json    │ For
│  │  ├─ et_instruct_164k_vid.json    │ Stage-3
│  │  └─ et_instruct_164k_meco.json  ─┘
├─ checkpoints
│  ├─ Phi-3-mini-4k-instruct
│  ├─ MiniCPM-V-2_6_QWen2
│  ├─ eva_vit_g.pth
│  └─ instruct_blip_vicuna7b_trimmed.pth
├─ meco
├─ scripts
└─ README.md
```
### Benchmarks
Please follow [ETBench](https://github.com/PolyU-ChenLab/ETBench) to download the ETBench benchmark, and follow [R2-Tuning](https://github.com/yeliudev/R2-Tuning) for the preparations of the Charades-STA and QVHighlights (QVH) benchmarks. The downloaded datasets should be placed in the `data` folder as well.

## 🔮 Training

Use the following commands to train MeCo. The default setting is to use 4 GPUs. You may modify `nproc_per_node`, `per_device_train_batch_size`, and `gradient_accumulation_steps` to keep the same global batch size if you have different device configurations.

```shell
# Stage-1
bash scripts/train_stage_1_phi3.sh # 4B
bash scripts/train_stage_1_minicpmv_qwen2.sh # 7B
# Stage-2
bash scripts/train_stage_2_phi3.sh [<path-to-stage-1-checkpoint>]
bash scripts/train_stage_2_minicpmv_qwen2.sh [<path-to-stage-1-checkpoint>]
# Stage-3
bash scripts/train_stage_3_phi3.sh [<path-to-stage-2-checkpoint>]
bash scripts/train_stage_3_minicpmv_qwen2.sh [<path-to-stage-2-checkpoint>]
```

The training logs and checkpoints will be saved in the `work_dirs` folder.

## 💻 Inference

Use the following commands to run inference on different benchmarks.

### E.T. Bench Evaluation

```shell
bash scripts/inference_etbench.sh [<path-to-checkpoint>] [<task>] [<pred-dir>] [<dtype>]
```

### Charades-STA QVH Evaluation

```shell
bash scripts/inference_charade_qvh.sh [<anno-path>] [<data-path>] [<path-to-checkpoint>] [<pred-path>] [<dtype>]
```

This will generate prediction files that can be evaluated using the corresponding evaluation scripts in the `tools` folder.

After running inference, you can compute evaluation metrics using the provided scripts.

### E.T. Bench Metrics

```shell
# Compute metrics for E.T. Bench predictions
python tools/compute_metrics.py <path-to-prediction-file-or-dir>
```

### Charades-STA QVH Metrics

```shell
# Compute metrics for Charades-STA QVH predictions  
python tools/compute_metrics_charades_qvh.py <path-to-prediction-file-or-dir>
```
## 🙏 Acknowledgments

This README is adapted from the [E.T. Chat documentation](https://github.com/PolyU-ChenLab/ETBench/blob/main/docs/MODEL.md).

This work is built upon the excellent foundations of:
- [E.T. Chat](https://github.com/PolyU-ChenLab/ETBench)
- [R2-Tuning](https://github.com/yeliudev/R2-Tuning)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [MomentDETR](https://github.com/jayleicn/moment_detr)
