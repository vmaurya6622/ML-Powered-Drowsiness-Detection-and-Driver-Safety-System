# Drowsiness Detection and Driver Safety System

## Overview
This project implements a **Drowsiness Detection and Driver Safety System** using **Arduino** and real-time facial monitoring techniques. The system aims to enhance road safety by identifying signs of driver fatigue and triggering alerts to prevent accidents caused by drowsy driving.

## Features
- Real-time facial monitoring using a camera.
- Drowsiness detection using eye aspect ratio and head pose analysis.
- Alerts to notify the driver when drowsiness is detected.
- Arduino integration for additional hardware control (e.g., triggering alarms).

## Project Structure
```
├── model.ipynb          # Main Jupyter Notebook with code implementation
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── Driver Drowsiness Dataset (DDD)/  # Sample images/screenshots (if applicable)
└── src/                 # Additional scripts and utilities
```

## Prerequisites
Ensure the following software and hardware are available:

### Software Requirements
1. Python 3.8 or higher
2. Jupyter Notebook
3. Libraries listed in `requirements.txt` (e.g., OpenCV, NumPy, dlib, imutils, etc.)
4. Arduino IDE (if using an Arduino for hardware integration)

### Hardware Requirements
1. A computer with a camera/webcam.
2. Arduino board (e.g., Arduino Uno) for hardware components.
3. Buzzer or LED for notifications (optional).

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/vmaurya6622/ML-Powered-Drowsiness-Detection-and-Driver-SafetySystem.git
   cd drowsiness-detection
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Configure the Arduino:
   - Upload the Arduino sketch provided in the repository (if applicable) using the Arduino IDE.
   - Connect the Arduino to your computer.

## Running the Project

### Step 1: Start the Jupyter Notebook
Launch Jupyter Notebook in the project directory:
```bash
jupyter notebook
```
Open `model.ipynb` and execute the cells sequentially.

### Step 2: Camera Access
Ensure your camera is connected and functional. The system will access the camera to capture real-time video frames.

### Step 3: Run Drowsiness Detection
The notebook will process the video feed, detect drowsiness based on facial landmarks, and trigger alerts if necessary.

### Step 4: (Optional) Integrate Arduino
If using an Arduino, ensure it is connected via USB. Alerts such as a buzzer sound or LED light will be triggered when drowsiness is detected.

## How It Works
1. **Face Detection**: The system detects the driver's face in real time using a pre-trained face detection model (e.g., Haar cascades or dlib).
2. **Eye Aspect Ratio (EAR)**: Calculates the EAR to monitor if the eyes are closed for an extended period.
3. **Head Pose Analysis**: Detects head tilts that indicate drowsiness.
4. **Alerts**: Triggers a warning sound or visual indicator if drowsiness is detected.

## Troubleshooting
- If the camera feed does not appear, verify your webcam permissions.
- Ensure all required libraries are installed using `requirements.txt`.
- If Arduino integration fails, check the COM port configuration and ensure the Arduino is properly connected.

## Acknowledgments
- OpenCV and dlib libraries for facial landmark detection.
- Arduino for hardware integration.
- Our professor for guidance and support during the project development.

---
