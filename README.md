# Abalone Age Prediction using Sequential Neural Network

A machine learning project that predicts the age (number of rings) of abalones using a Sequential deep learning model built with TensorFlow/Keras.

## 📋 Overview

Abalones are marine mollusks whose age is traditionally determined by cutting the shell, staining it, and counting the number of rings under a microscope. This project uses a neural network to predict the number of rings (and thus the age) based on easily measurable physical characteristics.

## 📊 Dataset

The **Abalone dataset** contains the following features:

| Feature | Description | Type |
|---------|-------------|------|
| Sex | M (Male), F (Female), I (Infant) | Categorical |
| Length | Longest shell measurement (mm) | Continuous |
| Diameter | Perpendicular to length (mm) | Continuous |
| Height | With meat in shell (mm) | Continuous |
| Whole Weight | Whole abalone (grams) | Continuous |
| Shucked Weight | Weight of meat (grams) | Continuous |
| Viscera Weight | Gut weight after bleeding (grams) | Continuous |
| Shell Weight | After being dried (grams) | Continuous |
| Rings | Target variable (+1.5 gives age in years) | Integer |

**Dataset Statistics:**
- Total samples: 4,177
- No missing values
- Sex distribution: Male (36.5%), Female (31.5%), Infant (32.0%)

## 🏗️ Model Architecture

The project uses a **Sequential Neural Network** with the following layers:

```python
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)  # Regression output for predicting rings
])
```

### Key Components:
- **Dense Layers**: Fully connected layers with ReLU activation
- **Batch Normalization**: Normalizes layer inputs for faster training
- **Dropout**: Regularization to prevent overfitting

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.8+ installed, then install the required dependencies:

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Sequential-Model-
   ```

2. Open the Jupyter notebook:
   ```bash
   jupyter notebook sequential-model-for-abalone-dataset.ipynb
   ```

## 📈 Project Workflow

### 1. Data Loading & Exploration
```python
# Load the Abalone dataset
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 
                'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings']
df = pd.read_csv('abalone.zip', names=column_names)
```

### 2. Data Preprocessing
- Check for missing values
- Encode categorical variables (one-hot encoding for 'Sex')
- Feature scaling using StandardScaler
- Train-test split

### 3. Model Training
```python
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
)
```

### 4. Evaluation
- Training and validation loss visualization
- Prediction vs actual comparison
- Performance metrics (MSE, MAE)

## 📊 Visualizations

The notebook includes various visualizations:
- Distribution of Sex categories (pie chart)
- Scatter plots of features vs Rings
- Correlation heatmaps
- Training history plots
- Prediction results

## 🎯 Results

The Sequential neural network model successfully predicts the number of rings (age) of abalones with reasonable accuracy based on their physical measurements.

## 📁 Project Structure

```
Sequential-Model-/
├── README.md                                    # Project documentation
├── sequential-model-for-abalone-dataset.ipynb  # Main Jupyter notebook
└── LICENSE                                      # License file
```

## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical operations
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Data preprocessing and train-test split
