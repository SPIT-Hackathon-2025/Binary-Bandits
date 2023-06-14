# Automatic Number Plate Recognition (ANPR) in MATLAB

This repository contains the implementation of an Automatic Number Plate Recognition (ANPR) system using MATLAB. The system utilizes image processing techniques to detect and recognize vehicle number plates from images or video streams. It can be used for various applications such as traffic monitoring, toll collection systems, parking lot management, and law enforcement.

## Table of Contents

- [Introduction](#introduction)
- [Methodology](#methodology)
- [Usage](#usage)
- [Results and Discussion](#results-and-discussion)
- [Implementation](#implementation)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The ANPR system is designed to capture images of vehicles, detect the number plates, and extract the characters from the plates using Optical Character Recognition (OCR) techniques. It involves several stages including image acquisition, pre-processing, plate detection, plate segmentation, character recognition, and output display. The system utilizes various image processing techniques such as resizing, noise removal, edge detection, morphological operations, and feature extraction to accurately identify and recognize number plates.

## Methodology

The ANPR system follows the following methodology:

1. **Image Acquisition:** The system acquires an image of a vehicle with a number plate using a camera or retrieves it from a dataset.
2. **Pre-processing:** The acquired image undergoes pre-processing operations such as resizing, noise removal, contrast enhancement, and image filtering to enhance its quality for further processing.
3. **Plate Detection:** An algorithm is applied to locate the number plate region in the pre-processed image. Techniques such as edge detection, morphological operations, and color thresholding can be used for plate detection.
4. **Plate Segmentation:** The number plate region is segmented to separate the characters from the background. This can be done using connected component analysis and contour analysis techniques.
5. **Character Recognition:** The segmented characters are recognized using OCR techniques such as template matching, feature extraction, and machine learning.
6. **Output Display:** The recognized characters are displayed as the final output, along with the original image and the location of the detected number plate.

## Usage

To use the ANPR system, follow these steps:

1. Clone the repository to your local machine.
2. Launch MATLAB and navigate to the project directory.
3. Modify the code as needed to specify the input image or video stream.
4. Run the MATLAB script to execute the ANPR system.
5. View the output, which will display the recognized characters and the processed image with the detected number plate highlighted.

## Results and Discussion

The ANPR system has been tested on various images of vehicles with different sizes and has shown good results in detecting and recognizing number plates. The system's accuracy, processing time, and detection rate can be evaluated by comparing the recognized characters with the ground truth data.

## Implementation

The implementation of the ANPR system is done in MATLAB (R2022b). The code consists of several MATLAB scripts and functions that perform specific tasks such as image pre-processing, plate detection, plate segmentation, and character recognition. These scripts can be executed sequentially to run the ANPR system.

## Dependencies

The ANPR system in MATLAB relies on the following dependencies:

- MATLAB (R2022b or higher)
- Image Processing Toolbox

Make sure you have MATLAB installed on your machine and the Image Processing Toolbox is available.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to submit a pull request.

## License

You are free to use, modify, and distribute the code for personal and commercial purposes.
