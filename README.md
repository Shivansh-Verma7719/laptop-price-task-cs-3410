# Laptop Price Predictor

This is a simple machine learning project that predicts laptop prices based on specs like CPU, RAM, storage, etc.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Put your training data in `data/train_data.csv`

3. Train the model:
```bash
python src/train_model.py --data_path data/train_data.csv
```

4. Make predictions:
```bash
python src/predict.py --model_path models/regression_model_final.pkl --data_path data/train_data.csv --metrics_output_path results/train_metrics.txt --predictions_output_path results/train_predictions.csv
```

## What It Does

- Trains different models (linear, ridge, lasso, polynomial) and picks the best one
- Handles data preprocessing automatically
- Saves the trained model and makes predictions
- Outputs predictions in a CSV file and some basic stats

## Files

- `src/train_model.py` - Training script
- `src/predict.py` - Prediction script
- `models/` - Saved trained models
- `results/` - Prediction results and metrics

I built this from scratch using gradient descent for the models and basic Python libraries.

