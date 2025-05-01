import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


train_data = pd.DataFrame(X_train, columns=[f'pixel_{i}' for i in range(X.shape[1])])
train_data['target'] = y_train
train_data.to_csv("train_data.csv", index=False)

def train_model(C=1.0, max_iter=300):
    with mlflow.start_run() as run:
        
        
        model = LogisticRegression(C=C, max_iter=max_iter, solver='lbfgs', multi_class='auto')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("model_type", "Logistic Regression")
        mlflow.log_param("dataset", "Digits Dataset")

        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        
        mlflow.log_artifact("train_data.csv")

        
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name="LogisticRegression_Digits"
        )

        print(f"Logged run {run.info.run_id} with Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

if __name__ == "__main__":
    train_model(C=1.0, max_iter=300)
    train_model(C=0.1, max_iter=500)
    train_model(C=2.0, max_iter=400)
