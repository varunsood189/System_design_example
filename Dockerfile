FROM python:3.9

WORKDIR /mlflow

RUN pip install mlflow

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/mlflow/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]

