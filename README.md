# SoccerSense ⚽🔍

![SoccerSense](https://i.pinimg.com/originals/bc/49/6b/bc496b9be3b3ef214da88ead21dc2e8a.gif)

SoccerSense is a comprehensive football analysis system that leverages machine learning, computer vision, and deep learning to extract meaningful insights from football match videos. Whether you're a beginner looking to get started with computer vision or an experienced engineer aiming to explore advanced techniques, SoccerSense offers a hands-on experience with state-of-the-art tools and methodologies.

## Overview

In this project, you will learn how to:

1. **Detect Objects with YOLOv8**: Utilize Ultralytics YOLOv8, a state-of-the-art object detector, to detect players, referees, and footballs in images and videos.
2. **Train Custom Object Detectors**: Fine-tune and train your own YOLO model on a custom dataset to enhance detection accuracy.
3. **Segment Players by T-Shirt Color**: Use KMeans clustering for pixel segmentation to accurately assign players to teams based on their t-shirt colors.
4. **Track Camera Movement**: Implement optical flow techniques to measure camera movement between frames, crucial for accurate player tracking.
5. **Perspective Transformation**: Apply OpenCV's perspective transformation to represent the scene's depth and perspective, allowing you to measure player movement in meters rather than pixels.
6. **Calculate Speed and Distance**: Measure players' speed and the distance covered on the field using advanced tracking techniques.

## Project Structure

- **`data/`**: Contains datasets used for training and evaluation.
- **`models/`**: Pre-trained models and custom-trained YOLO models.
- **`notebooks/`**: Jupyter notebooks for step-by-step guides on implementing each component of the project.
- **`src/`**: Core source code for the SoccerSense system.
- **`README.md`**: Project documentation and instructions.

## Datasets

This project utilizes the following datasets:

- **Kaggle Bundesliga Dataset**: [DFL Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/data?select=clips)
- **Roboflow Football Dataset**: [Football Players Detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)

## Installation

To run this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your-username/SoccerSense.git
cd SoccerSense
pip install -r requirements.txt
```

## Usage

1. **Object Detection**: Use the provided YOLOv8 models to detect objects in images or videos.
2. **Custom Training**: Follow the steps in the notebooks to train your custom object detector.
3. **Pixel Segmentation**: Apply KMeans clustering to segment players based on t-shirt colors.
4. **Camera Movement**: Use optical flow techniques to measure and compensate for camera movement.
5. **Perspective Transformation**: Transform video frames to accurately measure distances.
6. **Speed & Distance Calculation**: Calculate players' speed and distance covered using the provided tools.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.


## Acknowledgements

- The authors of the YOLOv8 model and the Ultralytics team.
- Kaggle and Roboflow for providing the datasets.
- The open-source community for the tools and libraries that made this project possible.
