# Water Potability Classification Using Neural Networks

## Projcet overview

The objective of this project was to evaluate whether artificial neural networks and machine learning models can classify water potability based on various physicochemical water quality measurements.

The project analyzes parameters such as pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes and turbidity.

Three main classification models are evaluated and compared:

- **Multilayer Perceptron (MLP) with architecture 64-32-16**,
- **Multilayer Perceptron (MLP) with architecture 256-128**,
- **Random Forest Classifier**.

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
- **Random Forest:** A Random Forest model consisting of 300 trees.

## Evaluation Metrics

Model performance is assessed using the following classification metrics:

- **Accuracy:** proportion of correctly classified water samples,
- **Recall:** proportion of potable samples correctly identified,
- **Specificity:** proportion of negative samples correctly identified,
- **Precision:** proportion of positive predictions that are correct,
- **F1-score:** harmonic mean of precision and recall.

# Results


| Metric | MLP (64-32-16) | MLP (256-128) | Random forest |
| --- | --- | --- | --- |
| Accuracy | 66,67% | 66,67% | 69,51% |
| Recall | 67,58% | 57,69% | 52,75% |
| Specificity | 66,13% | 71,94% | 79,35% |
| Precision | 53,95% | 54,69% | 60,00% |
| F1-score | 60,00% | 56,14% | 56,17% |

<p align="center">
  <img width="35%" alt="MLP (64-32-16) confusion matrix" src="https://github.com/3nrlque/Water-potability-classification/blob/main/Results/MLP_64-32-16_confusion_matrix.png"/> <img width="36.2%" alt="MLP (256-128) confusion matrix" src="https://github.com/3nrlque/Water-potability-classification/blob/main/Results/MLP_256-128_confusion_matrix.png"/> <img width="36.5%" alt="Random forest confusion matrix" src="https://github.com/3nrlque/Water-potability-classification/blob/main/Results/RF_confusion_matrix.png"/>
</p>

Based on the evaluation results, several key observations can be made regarding the performance of the models and the nature of the water quality dataset:

* The accuracy achieved by the evaluated models ranged from approximately 66.7% to 69.5%, indicating that predicting water potability based solely on standard physicochemical parameters is a challenging classification task. The relatively modest performance suggests that these features may not provide sufficient discriminatory information to reliably distinguish between potable and non-potable water samples.

* The Random Forest model achieved the highest accuracy (69.51%) and specificity (79.35%) among the evaluated models. This suggests that the ensemble-based approach was particularly effective at identifying non-potable water samples in the analyzed dataset. Its performance may be attributed to its ability to capture non-linear relationships between physicochemical parameters and model complex feature interactions without requiring feature scaling. In contrast, MLP performance can be influenced by factors such as feature scaling, weight initialization, and optimization settings.

* Random Forest achieved the highest overall accuracy and specificity, demonstrating a stronger ability to correctly identify non-potable water samples. In contrast, lighter MLP architectures, particularly the 64-32-16 configuration, achieved higher recall (67.58% compared to 52.75% for Random Forest), indicating a greater ability to identify potable water samples. These results highlight a trade-off between the models: Random Forest is more effective at recognizing non-potable water, whereas the MLP offers better sensitivity toward potable samples. Consequently, the choice of model should depend on the intended application and the relative importance of correctly identifying unsafe water versus minimizing the misclassification of potable water as non-potable.