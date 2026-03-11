# Breast Cancer Classification

## Project Structure
```
keras-classification-task.ipynb
deploy.py
model_weights.pkl
scaler_weights.pkl
```

## How to Run

### 1. Train the Model
Run all cells in `keras-classification-task.ipynb`

### 2. Start the API
```bash
uvicorn deploy:app --reload
```

### 3. Test
Open in browser: `http://localhost:8000/docs`

## Model
- Dataset: Breast Cancer Wisconsin
- Algorithm: Keras Neural Network
- Accuracy: 97%

## API Endpoints
- `GET /health` - Health check
- `POST /predict` - Predict single sample (30 features)
