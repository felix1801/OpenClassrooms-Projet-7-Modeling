import mlflow
import os

if __name__ == "__main__":
    os.system("mlflow server --host 127.0.0.1 --port 8080")