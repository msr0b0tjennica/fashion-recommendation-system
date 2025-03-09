# Fashion Recommendation System Using CNN

## Overview
This project implements a fashion recommendation system using Convolutional Neural Networks (CNN). It extracts deep learning-based features from fashion images and finds visually similar items based on cosine similarity.

## Features
- Extracts image features using a pre-trained VGG16 model.
- Finds similar images based on cosine similarity.
- Displays the recommended fashion items.

## Requirements
Make sure you have the following dependencies installed:

```bash
pip install tensorflow pillow matplotlib scipy
```

## Dataset
The dataset should be in a zipped format named `women-fashion.zip` and stored in Google Drive under `/content/drive/MyDrive/women-fashion.zip`. The extracted folder should contain images of fashion items.

## Usage
### Extracting Images
```python
from zipfile import ZipFile
import os

zip_file_path = '/content/drive/MyDrive/women-fashion.zip'
extraction_directory = '/content/drive/MyDrive/women-fashion/'

if not os.path.exists(extraction_directory):
    os.makedirs(extraction_directory)

with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_directory)
```

### Feature Extraction using VGG16
```python
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import numpy as np

base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)
```

### Recommendation Function
```python
from scipy.spatial.distance import cosine

def recommend_fashion_items_cnn(input_image_path, all_features, all_image_names, model, top_n=5):
    preprocessed_img = preprocess_image(input_image_path)
    input_features = extract_features(model, preprocessed_img)
    
    similarities = [1 - cosine(input_features, other_feature) for other_feature in all_features]
    similar_indices = np.argsort(similarities)[-top_n:]
    
    plt.figure(figsize=(15, 10))
    plt.subplot(1, top_n + 1, 1)
    plt.imshow(Image.open(input_image_path))
    plt.title("Input Image")
    plt.axis('off')
    
    for i, idx in enumerate(similar_indices[:top_n], start=1):
        image_path = os.path.join('/content/women_fashion/women fashion', all_image_names[idx])
        plt.subplot(1, top_n + 1, i + 1)
        plt.imshow(Image.open(image_path))
        plt.title(f"Recommendation {i}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
```

### Running the Recommendation System
```python
input_image_path = '/content/drive/MyDrive/women fashion/78f821325e689689d26b147649d22d1a.jpg'
recommend_fashion_items_cnn(input_image_path, all_features, list_paths_image, model, top_n=4)
```

## Results
The system displays an input image alongside visually similar fashion items, helping users find fashion recommendations based on CNN-based feature extraction.

## Author
Your Name

## License
This project is open-source under the MIT License.

