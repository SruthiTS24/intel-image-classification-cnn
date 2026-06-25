# Intel Image Classification using CNN

A Convolutional Neural Network built from scratch to classify natural scene images
into 6 categories using the Intel Image Classification dataset.

---

## Dataset

- **Source:** [Intel Image Classification — Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Classes:** Buildings, Forest, Glacier, Mountain, Sea, Street
- **Training images:** 14,034 | **Test images:** 3,000
- **Class distribution:** Balanced (2,191–2,512 images per class)

---

## Workflow

- Loaded dataset using `tf.keras.utils.image_dataset_from_directory`
- Resized all images to 150×150
- Split training folder 80/20 into train and validation sets (seed=42)
- Kept test folder completely separate for final unbiased evaluation
- Applied data augmentation (random flip, rotation, zoom) as model layers
- Applied pixel normalization (rescaling 0–255 to 0–1) as first model layer

---

## Model Architecture

| Layer | Details |
|---|---|
| Input | 150×150×3 |
| Data Augmentation | RandomFlip, RandomRotation, RandomZoom |
| Rescaling | 1./255 |
| Conv2D + MaxPooling | 32 filters, 3×3, ReLU |
| Conv2D + MaxPooling | 64 filters, 3×3, ReLU |
| Conv2D + MaxPooling | 128 filters, 3×3, ReLU |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dropout | 0.5 |
| Output Dense | 6 units, Softmax |

- **Total parameters:** 4,829,126
- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Callbacks:** EarlyStopping (patience=5), ModelCheckpoint

---

## Results

**Training stopped at epoch 13 (early stopping). Best weights restored from epoch 8 (lowest val_loss).**

| Dataset | Accuracy | Loss |
|---|---|---|
| Validation (best val_loss) | 79.90% | 0.546 |
| Test (final) | **80.90%** | 0.550 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Forest | 0.94 | 0.96 | **0.95** |
| Street | 0.82 | 0.85 | 0.84 |
| Sea | 0.74 | 0.86 | 0.79 |
| Buildings | 0.73 | 0.83 | 0.77 |
| Mountain | 0.78 | 0.72 | 0.75 |
| Glacier | 0.86 | 0.66 | 0.75 |
| **Macro Avg** | **0.81** | **0.81** | **0.81** |

Forest was the best-performing class due to its visually distinctive texture.
Glacier had the lowest recall — frequently confused with mountain and sea due to
shared visual features (rocky terrain and icy-blue tones).

---

## Tech Stack

- Python 3.x
- TensorFlow 2.21
- NumPy, Matplotlib
- Scikit-learn (evaluation metrics)
- Seaborn (confusion matrix)

---

## How to Run

1. Clone the repository
2. Download the dataset from Kaggle and place it as:
````````
intel_dataset/
├── seg_train/seg_train/

└── seg_test/seg_test/
````````
3. Run the notebook cells in order
4. To skip retraining, load the saved model directly:
```python
from tensorflow.keras.models import load_model
model = load_model('intel_cnn_model.keras')
```
