# Brain-Tumor-Detection
Deploy tensorflow keras Model on AWS 

# Brain Tumor Classification with CNN

![download](https://github.com/vaibhav030798/Brain-Tumor-Detection/assets/76680409/2fef9bed-0a02-48e1-b7de-0fd9cd378f26)


## Overview

This project focuses on training a convolutional neural network (CNN) to classify brain tumor images into two categories: "No Tumor" and "Tumor." It includes both the training code and a pre-trained model for making predictions.

## Table of Contents

- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Training](#training)
- [Testing](#testing)
- [Results](#results)
- [Server Requirements](#server-requirements)

## Dataset

We used the Brain Tumor Dataset for training and validation. The dataset can be found [here](link_to_dataset). Please refer to the dataset's documentation for details on its structure.

## Getting Started

To get started with this project, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/vaibhav030798/Brain-Tumor-Detection.git
   ```

## Install the necessary dependencies:
  ```bash
  pip install -r requirements.txt
  ```
## Training

To train the CNN model, run the following command:

```bash
python Brain_tumor_train.py
```

## Testing

To test the trained model on sample images, run the following command:

```bash
Predict_Tumor.py
```

## Server Requirements
To deploy this project on an AWS EC2 instance, you'll need:

  1. PuTTY for SSH communication
  2. WinSCP for file transfer
  3. A .ppk file for SSH authentication (generated from your .pem file)
