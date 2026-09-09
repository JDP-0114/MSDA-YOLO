# 🔍 MSDA-YOLO: Multispectral Object Detection for Road Scenes via Multi-Scale Feature Fusion and Dynamic Task Alignment

Welcome to the official implementation repository of our paper:

**“MSDA-YOLO: Multispectral Object Detection for Road Scenes via Multi-Scale Feature Fusion and Dynamic Task Alignment”**

This repository provides a PyTorch implementation of **MSDA-YOLO**, a lightweight visible-infrared multispectral object detector based on YOLOv11n.

## 🚀 Features

- **Multispectral Early Fusion:** Aligned RGB and infrared images are concatenated along the channel dimension to form a six-channel input.

- **Multi-Scale Feature Extraction:** MKEC enhances multi-scale feature representation through channel-wise heterogeneous grouped convolutions with different kernel sizes.

- **Enhanced Feature Fusion:** MBEP aggregates multi-source information at each pyramid level to strengthen cross-scale feature interaction.

- **Dynamic Task Alignment:** DTID constructs shared task-interactive features at each pyramid level and applies task-specific recalibration and dynamic regression alignment to improve the coordination between classification and localization.

- **Accuracy–Complexity Trade-off:** MSDA-YOLO improves detection accuracy while maintaining a relatively compact parameter scale, although the additional feature aggregation and dynamic alignment operations increase the computational cost compared with the YOLOv11n baseline.

## 📊 Performance

| Dataset | mAP50 | mAP50:95 |
|---|---:|---:|
| M3FD | 81.69% | 56.40% |
| FLIR_Aligned | 85.22% | 52.00% |

## 📥 Getting Started

### Installation

1. Clone this repository:

```bash
git clone https://github.com/JDP-0114/MSDA-YOLO.git
cd MSDA-YOLO
```

2. Create a Python environment:

```bash
conda create -n msda-yolo python=3.10 -y
conda activate msda-yolo
```

3. Install PyTorch:

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

4. Install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The detection head requires the compiled operators from the full MMCV
package. Verify the installation with:

```bash
python -c "from mmcv.ops import ModulatedDeformConv2d; print('MMCV ops OK')"
```

## 📂 Dataset Preparation

### M3FD Dataset

Download the processed M3FD dataset from Baidu Netdisk:

- **Download link:** [M3FD Dataset](https://pan.baidu.com/s/1KOlEDZAeiI6rhRFh3681yQ?pwd=0114)
- **Extraction code:** `0114`

After downloading and extracting the dataset, place it in:

```text
dataset/M3FD/
```

### FLIR_Aligned Dataset

Download the processed FLIR_Aligned dataset from Baidu Netdisk:

- **Download link:** [FLIR_Aligned Dataset](https://pan.baidu.com/s/1z7OO3TYCkfY9Op3rj-Q-WA?pwd=0114)
- **Extraction code:** `0114`

After downloading and extracting the dataset, place it in:

```text
dataset/FLIR_Aligned/
```

### Dataset Directory Structure

The complete dataset directory should be organized as follows:

```text
dataset/
├── M3FD/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── images_ir/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── FLIR_Aligned/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── images_ir/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── m3fd_data.yaml
└── flir_data.yaml
```

## 📚 Training

Before training, open `train.py` and select the dataset configuration.

For M3FD:

```python
data="dataset/m3fd_data.yaml"
```

For FLIR_Aligned:

```python
data="dataset/flir_data.yaml"
```

The default experimental settings are:

| Setting | Value |
|---|---:|
| Image size | 640 |
| Batch size | 8 |
| Epochs | 350 |
| Workers | 4 |
| Optimizer | SGD |
| Initial learning rate | 0.01 |
| Momentum | 0.937 |

Run the training script:

```bash
python train.py
```

Training results are saved in the `runs/train` directory.

The best and last checkpoints are saved in:

```text
runs/train/<experiment_name>/weights/
├── best.pt
└── last.pt
```

## 🔬 Testing

The trained MSDA-YOLO weights are provided in the `weights` directory:

```text
weights/
├── m3fd_best.pt
└── flir_best.pt
```

Before testing, make sure the dataset paths in `dataset/m3fd_data.yaml`
and `dataset/flir_data.yaml` point to the datasets on your machine.

For M3FD, use:

```python
from ultralytics import YOLOMM

model = YOLOMM("weights/m3fd_best.pt")

result = model.val(
    data="dataset/m3fd_data.yaml",
    split="test",
    imgsz=640,
    batch=8,
    workers=4,
    device=0,
    project="runs/M3FD/test",
    name="MSDA-YOLO",
)
```

For FLIR_Aligned, use:

```python
from ultralytics import YOLOMM

model = YOLOMM("weights/flir_best.pt")

result = model.val(
    data="dataset/flir_data.yaml",
    split="test",
    imgsz=640,
    batch=8,
    workers=4,
    device=0,
    project="runs/FLIR/test",
    name="MSDA-YOLO",
)
```

Alternatively, set `model_path` and `data` in `val.py`, then run:

```bash
python val.py
```

The validation results include:

- Precision
- Recall
- mAP50
- mAP50:95

The results are saved in the configured output directory:

```text
runs/M3FD/test/MSDA-YOLO/
# or
runs/FLIR/test/MSDA-YOLO/
```

## 📁 Model Configuration

The MSDA-YOLO model configuration file is located at:

```text
ultralytics/cfg/models/MSDA-YOLO.yaml
```

## ⚙️ Experimental Environment

Our experiments were conducted with the following environment:

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.2.2 |
| CUDA | 12.1 |
| Ultralytics | 8.3.9 |
| Input size | 640 × 640 |
| Batch size | 8 |
| Workers | 4 |

## 📖 Citation

The manuscript is currently under review. Citation information will be
updated after the paper is officially published.

## 📜 License

This project is released under the AGPL-3.0 License.

## 🙏 Acknowledgements

This project is built upon the Ultralytics YOLO framework. We sincerely thank the open-source community for their valuable contributions.
