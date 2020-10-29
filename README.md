## Introduction
The most advanced transistors in current high-end integrated circuits are the fin field-effect transistors (FinFETs). It is believed that the gate-all-around field effect transistors (GAAFETs) fabricated on vertical nanopillars will be the new architecture that can improve the performance and efficiency of future electronic devices. However, fabricating a patterned array of tall nanopillars is challenging as they collapse during fabrication. Here, we propose a fast neural network architecture to identify and count collapsed nanopillars from top down transmission electron microscope (TEM) images with high accuracy.

## Preparing training dataset
The location os nanopillars in TEM is identified and a small region is cropped around it and resized to 64x64 pixels (_./dataset/allImages.tar.gz_). Then this image is manually labelled as upright or collapsed. We labelled a total of 4537 nanopillars (_./dataset/labelledDataset.dat_) out of which 2155 were collapsed and 2382 were upright.

## Data augmentation
Data aurgmentation was performed by
1. Rotating each labelled image by 90, 180, and 270 degrees.
2. Flipping each image around the vertical axis.
After data augmentation the labelled dataset had 36296 images (17240 – collapsed, 19056 – upright). Finally, this dataset was partitioned into two sets – 90% of the labelled images were used for training and the remaining 10% were used for validation of the model.

## Model architectures
For training the labelled images were resized to 32x32 pixels. We trained three different model architectures to classify nanopillars as collapsed or upright.
1. A fully connected neural network (_DNN_) with four dense layers of size 512, 512, 256, and 2. The first 3 layers used ReLU activation, while the last layer used softmax activation.
2. A convolution neural network (_CNN_) with 2 convolution layers of size 5x5, followed by half-maxpooling. Following this the feature map was flattened and connected to three dense layers of size 256, 128, and 2. The convolution layers, and the first two layers used ReLU activation, while the last layer used softmax activation.
3. Transfer learning using _VGG16_ model _imagenet_ weights. Two dense layers of size 100 and 2 with ReLU and softmax activation respectively were appended at the end of convolution and maxpooling layers of VGG16 model. The training was done only on the last two layers.

## Results
1. DNN model

|       N = 36296      | Prediction              |  
|                      | Collapse   | Upright    |  
|-----------|----------|------------|------------|  
|   Actual  | Collapse | 17110      | 130        |  
|   Actual  | Upright  | 83         | 18973      |  

2. CNN model

| N = 36296 |          | Prediction | Prediction |
|           |          | Collapse   | Upright    |
|-----------|----------|------------|------------|
| Actual    | Collapse | 17195      | 45         |
| Actual    | Upright  | 11         | 19045      |

3. VGG16 model

| N = 36296 |          | Prediction | Prediction |
|           |          | Collapse   | Upright    |
|-----------|----------|------------|------------|
| Actual    | Collapse | 17225      | 15         |
| Actual    | Upright  | 8          | 19048      |

## Comparison to image processing
The image processing algorithm marks the pixels represented by nanopillars as 1 and the background as 0 to make a binary image. Then the aspect-ratio of the nanopillar, and its distance to the nearest nanopillar is calculated. If the aspect-ratio is more than 1.5 and the distance is more than 16 pixels, it is marked as a collased nanopillars.

## Citing this work
If you use these modules for your research work please cite the original paper as
........
