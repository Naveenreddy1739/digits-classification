# digits-classification
mlops digits classification project
# Digits Classification with Logistic Regression and MLflow

This project demonstrates how to build, evaluate, and track a machine learning model using the Digits dataset from `scikit-learn`. It uses Logistic Regression for multi-class classification and logs experiments using MLflow.

## Dataset

- **Source**: `sklearn.datasets.load_digits()`
- **Description**: 8x8 pixel grayscale images of handwritten digits (0 through 9).
- **Type**: Multi-class classification

##  Project Structure

- `train.py`: Main training script that loads the dataset, trains the model, evaluates it, and logs everything to MLflow.
- `train_data.csv`: Saved training data used for reproducibility.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

##  How to Run

1. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

2. **Start MLflow UI** (optional):

    ```bash
    mlflow ui
    ```

    Then open [http://localhost:5000](http://localhost:5000) to browse your experiments.

3. **Run the training script**:

    ```bash
    python train.py
    ```

    This will train and log multiple Logistic Regression models with different hyperparameters.

##  Features

- Uses `MLflow` to track:
  - Model parameters
  - Accuracy and F1 Score
  - Artifacts like the dataset
  - Serialized models with signature
- Modular and easy to adapt to other datasets/models


