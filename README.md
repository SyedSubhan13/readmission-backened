# Readmission Forecasting Backend

This is the backend API service for the Readmission Forecasting application. It provides endpoints for patient data management, model prediction, and model evaluation.

## Tech Stack

- FastAPI - Modern, fast web framework for building APIs
- Python 3.9+ - Programming language
- Scikit-learn - Machine learning library
- Pandas - Data manipulation
- Vercel - Deployment platform

## API Endpoints

The API provides the following endpoints:

### Health Check
- `GET /health` - Check API health and available models

### Models
- `GET /available-models` - Get list of available models
- `GET /models/{model_name}/performance` - Get model performance metrics
- `POST /test-model/{model_name}` - Test a model with validation data
- `POST /compare-models` - Compare predictions from multiple models

### Predictions
- `POST /predict-normalized/{model_name}` - Make predictions with normalized data

### Data
- `GET /visualization-data` - Get data for visualization purposes
- `POST /upload-data` - Upload a CSV file with patient data

## Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the development server:
   ```
   python main.py
   ```

3. Test the API:
   ```
   python test_api.py
   ```

## Folder Structure

- `app.py` - Main application code with API endpoints
- `main.py` - Entry point for the application
- `requirements.txt` - Python dependencies
- `vercel.json` - Vercel deployment configuration
- `models/` - Directory for ML model files
- `data/` - Directory for data files
- `model_report/` - Directory for model reports

## Vercel Deployment

This backend is configured for deployment on Vercel. The `vercel.json` file contains the necessary configuration.

## Environment Variables

- `PORT` - Port for the API server (default: 8000)
- `PYTHONPATH` - Python path (automatically set by Vercel) 