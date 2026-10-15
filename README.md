# Water Potability Classification Using Neural Networks

## Projcet overview

The objective of this project is to evaluate whether artificial neural networks and machine learning models can classify water potability based on various physicochemical water quality measurements.

The project analyzes parameters such as pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes and turbidity

Three main classification models are evaluated and compared:

- **Multilayer Perceptron (MLP) with architecture 64-32-16**
- **Multilayer Perceptron (MLP) with architecture 256-128**
- **Random Forest Classifier**

## Dataset

The experiments use the [Water Quality and Potability](https://www.kaggle.com/datasets/uom190346a/water-quality-and-potability/data) dataset. 

The original dataset is imbalanced, containing a higher number of samples labeled as non-potable (class 0) compared to potable water (class 1).

| Dataset Parameter | Value |
| --- | --- |
| Total samples | 3276 |
| Class 0 (Non-potable) | 1998 samples |
| Class 1 (Potable) | 1278 samples |

> **Dataset access:** Please consult the dataset provider's terms and conditions. This repository does not imply ownership of, or unrestricted redistribution rights to, the original water quality measurements.

## Methodology

### Data Preprocessing

- **Missing values imputation:** Missing cells in columns were filled using the mean value of the remaining entries in that specific column.
- **Data splitting:** The dataset was split into training (70%), validation (15%), and test (15%) sets.
- **Standardization:** Features were standarized using `StandardScaler` to have a mean of 0 and a standard deviation of 1.
- **Class balancing:** To address the class imbalance in the training set, the Synthetic Minority Over-sampling Technique (SMOTE) was applied, balancing the classes to an equal number of samples.

Class distibution before and after SMOTE:

<p align="center">
<img width="49%" alt="Class distibution before SMOTE" src="https://github.com/3nrlque/Water-potability-classification/blob/main/Results/class_distribution_before_SMOTE.png" /> <img width="49.1%" alt="Class distibution after SMOTE" src="https://github.com/3nrlque/Water-potability-classification/blob/main/Results/class_distribution_after_SMOTE.png" />
</p>

### Model Configurations

- **MLP (64-32-16):** Fully connected feedforward network with hidden layers of sizes 64, 32, and 16, utilizing ReLU activation, Adam solver, early stopping, and a maximum of 500 iterations.
- **MLP (256-128):** Fully connected network with larger hidden layers (256 and 128), ReLU activation, Adam solver, and early stopping.
- **Random Forest:** Ensemble model configured with 300 estimators.

## Evaluation Metrics

Model performance is assessed using the following classification metrics:

- **Accuracy:** proportion of correctly classified participants.
- **Recall:** proportion of participants with ADHD correctly identified.
- **Specificity:** proportion of control participants correctly identified.
- **Precision:** proportion of positive predictions that are correct.
- **F1-score:** harmonic mean of precision and recall.

# Results

The overall highest accuracy was achieved by the Random Forest classifier, which reached an accuracy of 69.51%.

**MLP (64-32-16):**
Achieved an accuracy of 66.67%, recall of 67.58%, specificity of 66.13%, precision of 53.95%, and an F1-score of 60.00%.

<p align="center">
  <img width="60%" alt="MLP (64-32-16) confusion matrix" src="https://github.com/3nrlque/Water-potability-classification/blob/main/Results/MLP_64-32-16_confusion_matrix.png"/>
</p>

**MLP (256-128):**
Achieved an accuracy of 66.67%, recall of 57.69%, specificity of 71.94%, precision of 54.69%, and an F1-score of 56.14%.

<p align="center">
  <img width="60%" alt="MLP (256-128) confusion matrix" src="https://github.com/3nrlque/Water-potability-classification/blob/main/Results/MLP_256-128_confusion_matrix.png"/>
</p>

**Random Forest:**
Achieved an accuracy of 69.51%, recall of 52.75%, specificity of 79.35%, precision of 60.00%, and an F1-score of 56.17%.

<p align="center">
  <img width="60%" alt="Random forest confusion matrix" src="https://github.com/3nrlque/Water-potability-classification/blob/main/Results/RF_confusion_matrix.png"/>
</p>

While the Random Forest model provided the highest overall precision and accuracy, the lighter MLP architecture (64-32-16) achieved a higher recall score, indicating better sensitivity toward detecting potable water samples.