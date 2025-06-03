#  🧠 Brain Cancer Classification Using Deep Learning (VGG16 & ResNet50)
This project leverages transfer learning using VGG16 and ResNet50 to classify brain tumors from MRI images into four categories: glioma, meningioma, pituitary, and notumor. The goal is to assist in early diagnosis and improve accuracy in brain cancer detection.

## 📂 Dataset Structure
The dataset should follow this directory structure:


#### Brain_Cancer_Detection/
#### ├── Training/
#### │   ├── glioma/
#### │   ├── meningioma/
#### │   ├── pituitary/
#### │   └── notumor/
#### └── Testing/
####     ├── glioma/
####     ├── meningioma/
####     ├── pituitary/
####     └── notumor/

Each subfolder contains labeled MRI brain scan images of the respective tumor type.


## ✅ Features
## Transfer learning using VGG16 and ResNet50

Fine-tuning with dropout regularization and dense layers

Image preprocessing and augmentation with ImageDataGenerator

Multi-class classification using softmax activation

## Model evaluation:

Accuracy

Precision, Recall, F1-score

Confusion Matrix

ROC-AUC Curve

Log Loss

Visualization of predictions and evaluation metrics

Image-level prediction support with confidence score

## 🧪 Dependencies
Install required Python packages:

>> pip install tensorflow numpy matplotlib seaborn scikit-learn keras

## 🚀 How to Run
Update dataset paths
Modify these lines to point to your dataset in the notebook:

>> train_dir = '/path/to/Training'
>> test_dir = '/path/to/Testing'

## Train the models

The models are trained with 10 epochs using transfer learning:

>> model.fit(train_generator,validation_data=val_generator, epochs=10)

## Evaluate performance

## After training, evaluate using:

Classification report

Confusion matrix

ROC curves

Log loss

## Predict new images
### Predict MRI images using:

>> predict_image('/path/to/image.jpg')

## 📊 Sample Results
Replace with actual values after training:

Model	Accuracy	Precision	Recall	F1-Score	Log Loss
VGG16	95.2%	0.95	0.95	0.95	0.12
ResNet50	96.7%	0.96	0.96	0.96	0.09

Note: Your results may vary depending on hardware, batch size, and training dataset variations.

## 📷 Sample Prediction Output

>> Prediction: pituitary (97.52%)

## 💾 Model Files
Best-performing models are saved at:


Saved_models/
├── VGG16_Best_Model.h5
└── resnet_best_model.h5

## 📝 License
This project is intended for educational and research purposes only. Please credit appropriately if you use it in any publications or derivative works.
