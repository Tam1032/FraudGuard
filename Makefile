PY ?= python

.PHONY: install train drift serve test smoke mlflow docker

install:
	$(PY) -m pip install -r requirements.txt

train:          ## needs data/raw/train_transaction.csv from Kaggle
	$(PY) -m fraudguard.train

drift:
	$(PY) -m fraudguard.drift

serve:
	uvicorn fraudguard.api:app --port 8000

test:
	$(PY) -m pytest -q tests

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Whole pipeline on fake data, to check the setup works before downloading Kaggle data.
smoke:
	$(PY) scripts/make_synthetic.py --rows 60000 --out data/raw/train_transaction.csv
	N_ESTIMATORS=300 $(PY) -m fraudguard.train
	$(PY) -m fraudguard.drift
	$(PY) -m pytest -q tests

docker:
	docker compose build
