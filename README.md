# Lane Detection Using Computer Vision

This project uses computer vision techniques to detect lane markings from road videos. The system currently supports detection of straight lane lines, and it is being extended to handle curved lanes as well.

## 🔍 Overview

The goal of this project is to process driving videos to detect lane lines in real-time. It uses edge detection, region of interest masking, and Hough Line Transform to identify lane markings and overlay them on the original video frames.

![Demo](GIF/output_lane_detection-ezgif.com-video-to-gif-converter.gif)

## ⚙️ Technologies Used

- **Python**
- **OpenCV**
- **NumPy**
- **Math**
- **Copy (for deep copy of previous states)**

## 🧠 Features

- Converts input frames to grayscale and applies Gaussian blur for noise reduction.
- Applies binary thresholding to highlight lane features.
- Defines a region of interest using a dynamic polygon to isolate road area.
- Uses Hough Line Transform for line detection.
- Differentiates between solid and dashed lines using slope-based filtering.
- Writes the output as an annotated video with overlayed lane lines.
- Automatically adjusts to different video resolutions.
- 🔜 **Coming Soon:** Curved lane detection using polynomial fitting or sliding window method.



