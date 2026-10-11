import fastapi
import mlflow
import numpy
import pandas
import sklearn

print("MLOps environment is ready!")
print(f"Pandas: {pandas.__version__}")
print(f"NumPy: {numpy.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"MLflow: {mlflow.__version__}")
print(f"FastAPI: {fastapi.__version__}")