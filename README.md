# 📊 MNIST Clustering Project

## 🔍 Overview

This project explores clustering techniques to classify and group similar elements within the MNIST dataset. The dataset comprises 70,000 grayscale images of handwritten digits, each sized 28×28 pixels. These images are divided into:
- Training Set: 60,000 images.
- Test Set: 10,000 images.

Each image is represented as a 784-dimensional vector of real values between 0 (black) and 1 (white). The goal is to cluster these images based on similarity using various algorithms and analyze the impact of dimensionality reduction.
<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/4b470221-6aee-439e-ac17-16eb093d02e5" alt="image">
</div>


## ⚙️ Clustering Techniques

The following clustering methods are used:

1. Mixture of Gaussians:
    - Diagonal covariance (Gaussian Naive Bayes with latent class labels).
    - Varies the number of clusters (‘k’) between 5 and 15.
2. Mean Shift:
    - Varies kernel width.
3. Normalized Cut:
    - Varies the number of clusters (‘k’) between 5 and 15.
  
<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/6ec1535b-d889-4dcc-a905-02e8e8f66926" alt="image">
</div>
## 📉 Dimensionality Reduction

To improve performance and analyze the effect of dimensionality, Principal Component Analysis (PCA) is applied, reducing dimensionality from 2 to 200. The impact on accuracy and learning time is evaluated.

<div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/d11dde8d-5165-45db-825d-a9a34f399360" alt="image">
</div>

## 🚀 Usage Instructions

### Requirements

Install the necessary Python libraries:
```bash
pip install numpy scikit-learn matplotlib
```

### Running the Clustering

1. Download Dataset:
Run DownloadData.py to save the dataset locally.
```bash
python DownloadData.py
```
2. Run Clustering:
Execute the clustering script with your desired method and parameters. Modify the number of clusters (‘k’) or kernel width for analysis.


## 📊 Evaluation Metrics

- Rand Index: Evaluates clustering accuracy by comparing clusters with ground truth.
- Learning Time: Measures computational efficiency across varying dimensions.


## 📂 References

- MNIST Dataset: Yann LeCun’s Website
- PCA and Clustering techniques from scikit-learn documentation.

