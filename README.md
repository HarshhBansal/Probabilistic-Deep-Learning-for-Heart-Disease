# Heart Disease Prediction using Deep Learning

## 📌 Project Overview

This project focuses on predicting the presence of heart disease using clinical (tabular) patient data and deep learning techniques. The main goal is to explore how neural networks—especially convolution-based models—can be applied to structured medical data and to analyze their behavior, strengths, and limitations.

The repository contains **two experimental approaches**:

1. A **Conv1D-based neural network** for heart disease classification.
2. A **probabilistic Conv2D-based model** using TensorFlow Probability to model prediction uncertainty.

> ⚠️ Note: Convolutional layers are traditionally designed for spatial data (images). In this project, they are intentionally applied to tabular data as an experiment. This reshaping does **not** introduce real spatial meaning and is mainly for research and learning purposes.


## 🧠 Dataset

* **Name:** Heart Disease (Cleveland + Hungary)
* **Type:** Tabular clinical dataset
* **Target Variable:** `target`

  * `0` → No heart disease
  * `1` → Presence of heart disease

### Features Include:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol
* Fasting blood sugar
* ECG results
* Maximum heart rate
* Exercise-induced angina
* And other clinical attributes

## ⚙️ Project Workflow

1. Load and inspect the dataset
2. Separate features (`X`) and target (`y`)
3. Split data into training and testing sets (80/20)
4. Standardize features using `StandardScaler`
5. Reshape tabular data to fit CNN input requirements
6. Build and train neural network models
7. Evaluate performance using accuracy

## 🧪 Model 1: Conv1D-based Neural Network (`heartdisease.py`)

### Architecture

* Conv1D (32 filters, kernel size = 3, ReLU)
* MaxPooling1D
* Flatten
* Dense (1 neuron, linear activation, L2 regularization)

### Key Points

* Each clinical feature is treated as a **sequence element**
* Conv1D slides across feature indices (not time or space)
* Binary cross-entropy loss is used
* Predictions are rounded to obtain class labels

### Result

* **Accuracy:** ~78–80% (varies slightly by run)

## 🧪 Model 2: Probabilistic Conv2D Model (`test.py`)

### Motivation

This model explores **uncertainty-aware predictions** using TensorFlow Probability (TFP) and a Bernoulli distribution.

### Architecture

* Conv2D
* MaxPooling2D
* Flatten
* Dense (64, ReLU)
* Dense (1 logit output)

### Key Concepts

* Tabular data is reshaped into a **fake 4D tensor** to satisfy Conv2D input
* No real spatial relationship exists between features
* Output logits are converted into probabilities using `sigmoid`
* A **Bernoulli distribution** models binary outcomes
* **Negative Log-Likelihood (NLL)** is used as the loss function

### Important Note ⚠️

* Class labels are **randomly sampled** from predicted probabilities
* This makes accuracy **non-deterministic**
* Same input may yield different outputs across runs

### Result

* **Accuracy:** ~0.78–0.79 (example run)


## 📈 Evaluation Metric

* **Accuracy Score** from `sklearn.metrics`
* Used only for demonstration
* Not sufficient alone for medical decision-making
  
## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirement.txt
```

**Main Libraries:**

* TensorFlow / Keras
* TensorFlow Probability
* NumPy
* Pandas
* Scikit-learn

## 🚀 How to Run

### Train Conv1D Model

```bash
python heartdisease.py
```

### Train Probabilistic Conv2D Model

```bash
python test.py
```

## 🧩 Key Learnings

* CNNs can technically process tabular data, but may not be optimal
* Dense layers handle feature interactions better for structured data
* Probabilistic modeling provides uncertainty but must be carefully wired
* Random sampling from probability distributions affects reproducibility

# Future Improvements

* Replace CNNs with ML models suited for tabular data (XGBoost, Random Forest)
* Use deterministic thresholding instead of random sampling
* Add evaluation metrics like ROC-AUC, Precision, Recall, F1-score
* Perform cross-validation
* Improve probabilistic integration inside the model graph

## 📜 Disclaimer

This project is for **educational and experimental purposes only**.
It is **not intended for real-world medical diagnosis**.


## 👤 Author

**Harsh Bansal**
AI/ML Developer

If you find this project helpful, feel free to ⭐ the repository!
