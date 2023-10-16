# YOLOv8-DeepSort-ByteTrack-PyQt-GUI
a GUI application, which uses YOLOv8 for  Object Detection/Tracking, Human Pose Estimation/Tracking from images, videos or camera. 

All python scripts performing detection, pose and segmentation using the YOLOv8 model in ONNX.

![GUI](./data/ui.png)

Supported AI tasks:
- [x] Detection
- [x] Pose Estimation
- [x] Segmentation

Supported Models:
- [x] YOLOv8n
- [x] YOLOv8s
- [x] YOLOv8m
- [x] YOLOv8l
- [x] YOLOv8x

Supported Trackers:
- [x] DeepSort
- [x] ByteTrack

Supported Input Sources:
- [x] local files: images or videos
- [x] Camera
- [x] RTSP-Stream

## Install

Install required packages with pip:

```shell
pip install -r requirements.txt
```

or with conda:

```shell
conda env create -f environment.yml

# activate the conda environment
conda activate yolov8_gui
```

## Download weights

Download the model weights：

``````shell
python download_weights.py
``````

The model files are saved in the **weights/** folder.

## Run

```shell
python main.py
```

