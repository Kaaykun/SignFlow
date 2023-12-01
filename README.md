# Sign Flow - A Sign Language Detection Application

![SignFlow Logo](frontend/SignFlowLogo.jpg)

This repository contains a sign language detection application built using Python. The application utilizes the Streamlit framework for the user interface, OpenCV for video processing, and TensorFlow/Keras for LSTM-based sign language prediction.

Checkout our app on Streamlit:
https://signflow.streamlit.app/

## Table of Contents
- [Files](#files)
- [Usage](#usage)
- [Features](#features)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [License](#license)

## Files

**Dataset**: https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed

1. **preprocessor.py**: 
   Contains functions for sampling frames from videos, creating feature matrices, and generating processed videos.

2. **registry.py**: 
   Includes functions for generating processed videos, CSV files, drawing landmarks, and saving/loading models.

3. **model.py**: 
   Defines the LSTM model architecture and functions for training the model.

4. **main.py**: 
   The main script that orchestrates the data processing, model training, and saving.

5. **app.py**: 
   The Streamlit application with two pages - one for uploading a video for sign detection and another for real-time sign detection from a webcam stream.

## Usage

### 1. Video Upload Page

- Upload a video file for sign language detection.
- The application will predict the sign language associated with the uploaded video.

### 2. Real-Time Detection Page

- Use a webcam for real-time sign language detection.
- The application displays the detected signs in real-time.

## Features

- Sign language detection using LSTM models.
- Two modes: video upload for single predictions and real-time webcam detection.
- User-friendly Streamlit interface.

## Dependencies

- TensorFlow
- OpenCV
- Streamlit
- mediapipe
- scikit-learn
- pandas
- numpy
- av
- streamlit_webrtc

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/sign-language-detection.git
   cd sign-language-detection

2. Navigate to the project directory:
   cd SignFlow

3. Install the required dependencies:
   pip install -r requirements.txt

## License
This app is not licensed.

