# TBBR-YOLOv11: Dataset Conversion and Inference Pipeline


This repository provides all the tools and scripts needed to adapt the [TBBR (Thermal Bridges on Building Rooftops) dataset](https://github.com/Helmholtz-AI-Energy/TBBRDet) that was originally designed to work with MMDetection and Detectron2, for use with Ultralytics YOLO v11, using the new `.txt` annotation format.


## Features

* **COCO to YOLO Conversion**: Scripts to convert the original TBBR dataset (COCO format) to the YOLO format required by Ultralytics v11, including image and annotation processing.
* **Model Adaptation to NCNN**: A pipeline to export trained YOLO models to the NCNN format, enabling efficient inference and benchmarking on edge devices such as the Raspberry Pi 5.
* **Image Conversion Utility**: Tools to convert your own RGB and thermal images (captured with a H30T thermal camera on a DJI Matrice 350RTK drone) into the 4-channel TIFF format expected by the YOLO model.
* **Ready for Research**: All scripts and utilities were developed as part of a Bachelor's thesis and are made public to help others work with the TBBR dataset using the Ultralytics framework quickly and efficiently.



## Folder Structure

### Original Data (COCO format)
Place your original dataset files inside the `data/` directory, following the original TBBR structure:

```
data/
├── train/
│   ├── Flug1_100-104Media_coco.json
│   └── images/
│       ├── Flug1_100Media/
│       │   ├── DJI_XXXX_R.npy
│       │   └── ...
│       ├── Flug1_104Media/
│       │   ├── DJI_XXXX_R.npy
│       │   └── ...
│       └── ...
├── test/
│   ├── Flug1_105Media_coco.json
│   └── images/
│       └── Flug1_105Media/
│           ├── DJI_XXXX_R.npy
│           └── ...
```

### Adapted YOLO Dataset
The converted YOLO dataset will be placed inside the `data/` directory as well, following this structure:

```
data/
└── thermal_yolo/
	├── dataset.yaml
	├── train/
	│   ├── images/
	│   └── labels/
	└── val/
		├── images/
		└── labels/
```

## Usage

1. **Convert the dataset:** Use the provided notebooks/scripts to convert the original COCO annotations and images to YOLO format.
2. **Train with YOLOv11:** Train your model using Ultralytics YOLO v11 on the adapted dataset.
3. **Export to NCNN:** Use the pipeline to export your trained model to NCNN for edge inference and benchmarking.
4. **Image conversion:** Convert your own RGB + thermal images to the required TIFF format using the provided utility.

## Pretrained Models

The models trained for the thesis (including YOLOv11 large, nano, and NCNN versions) are available in the [GitHub Releases section](../../releases).

## About

This repository was developed as part of a Bachelor's thesis and is released to help the community work with the TBBR dataset and Ultralytics YOLO framework more easily.
