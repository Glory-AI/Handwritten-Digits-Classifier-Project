# Handwritten Character Classifier 

This project demonstrates a deep learning prototype for recognizing handwritten characters, serving as a proof of concept for an Optical Character Recognition (OCR) system. Using the MNIST dataset, the project explores key PyTorch workflows such as data loading, image transformations, tensor operations, and neural network design with layers like Flatten, Softmax, and Pooling.

### Project Overview

This project builds on foundational deep learning concepts for computer vision.
It implements a neural network that classifies handwritten digits (0–9).

### Objectives

* Prototype a character recognition model using the MNIST handwritten digits dataset.

* Explore PyTorch tensor frameworks, DataLoader pipelines, and neural network layers.

* Understand and implement Flatten, Softmax, ReLU, and Pooling mechanisms.

* Preprocess and visualize data effectively for training and testing.

### Technologies and Frameworks

* Language: Python 

* Deep Learning Framework: PyTorch

* Dataset: MNIST (via torchvision.datasets)

* Visualization: Matplotlib

( Libraries: Torchvision, NumPy, Torch, Matplotlib

#### Getting Started

Follow these steps to set up and run the project locally.

* Clone the Repository
```
git clone https://github.com/Glory-AI/Handwritten-Digits-Classifier-Project.git
cd Handwritten-Digits-Classifier-Project
```
* Create a Virtual Environment (optional but recommended)
```
python -m venv venv
source venv/bin/activate      # For Linux/Mac
venv\Scripts\activate         # For Windows
```

* Install Dependencies
```
pip install -r requirements.txt
```
* Or manually install:
```
pip install torch torchvision matplotlib numpy
```
* Run the Notebook: handwritten_classifier.ipynb

### Workflow Summary

#### 1. Data Loading & Exploration

* Loaded training and test data using torchvision.datasets.MNIST.

* Created DataLoader objects for efficient batching and shuffling.

* Explored dataset properties — shape, size, and class distribution.

* Visualized sample images using matplotlib.pyplot.imshow().

#### 2. Data Transformation

* Applied transformations using torchvision.transforms.

* Converted images to tensors using .ToTensor() for PyTorch compatibility.

* Justified flattening: Neural networks expect 1D vector inputs — flattening converts (batch_size, height, width, channels) → (batch_size, height × width × channels) to allow processing by Linear layers.

#### 3. Model Architecture

The network consists of:

* Flatten Layer: Converts 2D image data into 1D tensors.

* Fully Connected Layers (Linear): Perform pattern extraction and prediction.

* ReLU Activation: Adds non-linearity.

* Softmax Output: Converts logits into probability distributions.
 
 

#### 4. Training & Evaluation

* Used CrossEntropyLoss for measuring performance.

* Optimized using SGD or Adam optimizers.

* Trained for multiple epochs and evaluated accuracy on the test dataset.


### Key Learnings

* Mastered tensor operations and data pipeline setup in PyTorch.

* Understood Flatten, ReLU, Pooling, and Softmax layers conceptually and practically.

* Gained experience in image data representation and batch processing.

* Learned the importance of preprocessing and normalization in neural network training.


### Future Improvements

* Extend the model to recognize letters and symbols for full OCR systems.
* Deploy the trained model using Flask or FastAPI for live inference.




