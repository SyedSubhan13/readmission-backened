"""
FastAPI server for readmission prediction API.
Includes functionality to load local models and data files.
Modified to handle normalized data inputs.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import json
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle
from datetime import datetime
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import logging
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model definitions
class PatientData(BaseModel):
    """Pydantic model for normalized patient data input"""
    encounter_id: Optional[str] = None
    patient_nbr: Optional[str] = None
    time_in_hospital: Optional[float] = None
    num_lab_procedures: Optional[float] = None
    num_procedures: Optional[float] = None
    num_medications: Optional[float] = None
    number_diagnoses: Optional[float] = None
    insulin_No: Optional[float] = None
    diag_3_grouped_other: Optional[float] = None
    time_in_hospital_x_num_medications: Optional[float] = None
    readmitted: Optional[float] = None

    class Config:
        schema_extra = {
            "example": {
                "encounter_id": "2278392",
                "patient_nbr": "8222157",
                "time_in_hospital": -1.13,
                "num_lab_procedures": -0.10,
                "num_procedures": -0.78,
                "num_medications": -1.84,
                "number_diagnoses": -3.32,
                "insulin_No": 1.0,
                "diag_3_grouped_other": 1.0,
                "time_in_hospital_x_num_medications": 2.10
            }
        }

class BatchPredictionRequest(BaseModel):
    data: List[PatientData]

app = FastAPI(title="Readmission Prediction API")

# Configure CORS - Updated to be more permissive for development
origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    # Add development origin
    "http://localhost:5173",  # Vite default port
    "http://127.0.0.1:5173",
    # Allow all origins in production
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Model and data paths
PROJECT_ROOT = Path(__file__).parent.resolve()  # Get absolute path
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
VISUALIZATION_DATA_PATH = DATA_DIR / "eda.csv"
TEST_DATA_PATH = DATA_DIR / "feature_eng.csv"

# Log the paths for debugging
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Model directory: {MODEL_DIR}")
logger.info(f"Data directory: {DATA_DIR}")

def load_model(model_type: str, model_path: Optional[str] = None) -> Any:
    """Load a model from the specified path."""
    try:
        if model_path:
            model_file = Path(model_path)
        else:
            # Convert model_type to lowercase to match actual filenames
            model_type = model_type.lower()
            
            # Handle different file extensions based on model type
            if model_type == 'mlp':
                model_file = MODEL_DIR / f"{model_type}.h5"
            else:
                model_file = MODEL_DIR / f"{model_type}.pkl"
        
        logger.info(f"Attempting to load model from: {model_file}")
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found at {model_file}")
        
        try:
            if model_file.suffix == '.h5':
                # Import tensorflow only when needed
                import tensorflow as tf
                model = tf.keras.models.load_model(model_file)
            else:
                try:
                    model = joblib.load(model_file)
                except:
                    model = pickle.load(open(model_file, 'rb'))
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
        
        return model, str(model_file)
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Add a root route for API validation
@app.get("/")
async def root():
    """Root endpoint to verify API is running"""
    return {
        "message": "Readmission Forecasting API is running",
        "version": "1.0.0",
        "status": "ok",
        "endpoints": {
            "health": "/health",
            "models": "/available-models",
            "predictions": "/predict-normalized/{model_name}",
            "data": "/visualization-data",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Check API health and available models"""
    try:
        # Check if model directory exists and contains models
        models_available = [f.stem for f in MODEL_DIR.glob("*.pkl")]
        
        # Check if data directory exists
        data_files = [f.name for f in DATA_DIR.glob("*.*")]
        
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "modelDir": str(MODEL_DIR),
            "dataDir": str(DATA_DIR),
            "modelsAvailable": models_available,
            "dataFiles": data_files
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available-models")
async def get_available_models():
    """Get list of available models in the model directory"""
    try:
        models = []
        # Look for both .pkl and .h5 files
        for model_file in MODEL_DIR.glob("*.[ph][k5][l5]"):
            models.append({
                "name": model_file.stem,
                "path": str(model_file),
                "size": model_file.stat().st_size,
                "lastModified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
            })
        
        return {
            "models": models,
            "count": len(models),
            "modelDirectory": str(MODEL_DIR)
        }
    except Exception as e:
        logger.error(f"Failed to get available models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualization-data")
async def get_visualization_data(
    limit: int = Query(1000, description="Number of records to return", ge=1, le=10000),
    include_stats: bool = Query(True, description="Whether to include column statistics")
):
    """Get data for visualization purposes with pagination and optional stats"""
    try:
        logger.info(f"Loading visualization data with limit: {limit}")
        
        if not VISUALIZATION_DATA_PATH.exists():
            raise FileNotFoundError(f"Visualization data file not found at {VISUALIZATION_DATA_PATH}")
        
        # Read the data efficiently
        df = pd.read_csv(VISUALIZATION_DATA_PATH)
        total_records = len(df)
        logger.info(f"Loaded DataFrame with shape: {df.shape}")
        
        # Apply limit
        if limit and limit < total_records:
            df = df.sample(limit) if limit > 0 else df
        
        # Calculate statistics if needed
        stats = None
        if include_stats:
            stats = {
                "numeric": df.describe().to_dict(),
                "counts": {col: df[col].value_counts().to_dict() for col in df.select_dtypes(include=['object', 'category']).columns}
            }
        
        # Convert to records format (list of dictionaries)
        records = df.to_dict(orient='records')
        
        return {
            "data": records,
            "total": total_records,
            "returned": len(records),
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting visualization data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """Upload a CSV file with data and save it to the data directory"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Determine the output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = file.filename
        filename_base, ext = os.path.splitext(original_filename)
        
        # Only accept CSV files
        if ext.lower() != '.csv':
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")
        
        # Create a timestamped filename
        output_file = DATA_DIR / f"{filename_base}_{timestamp}{ext}"
        
        # Save the uploaded file
        with open(output_file, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Load and validate the data
        try:
            df = pd.read_csv(output_file)
            rows, cols = df.shape
            
            return {
                "status": "success",
                "filename": output_file.name,
                "path": str(output_file),
                "rows": rows,
                "columns": cols,
                "columnNames": df.columns.tolist(),
                "dataTypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "timestamp": timestamp
            }
        except Exception as e:
            # If there's an error loading the CSV, remove the file and raise exception
            os.remove(output_file)
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error uploading data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return metrics"""
    try:
        # Start time
        start_time = time.time()
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        # Determine if this is a multiclass problem
        is_multiclass = len(np.unique(y_test)) > 2
        
        # Get probability predictions if possible
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)
                # For binary classification, take the positive class probability
                if proba.shape[1] == 2:
                    y_pred_proba = proba[:, 1]
                else:
                    # For multiclass, we'll use the max probability
                    y_pred_proba = np.max(proba, axis=1)
        except:
            # For models that don't support predict_proba
            logger.warning("Model doesn't support predict_proba")
            y_pred_proba = y_pred
        
        # Calculate metrics with appropriate average parameter for multiclass
        average_param = 'weighted' if is_multiclass else 'binary'
        logger.info(f"Using average='{average_param}' for metrics calculation (multiclass={is_multiclass})")
        
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0, average=average_param)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0, average=average_param)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0, average=average_param))
        }
        
        # Add ROC AUC only for binary classification
        if not is_multiclass and y_pred_proba is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # For binary classification, use the standard format
        if not is_multiclass:
            cm_dict = {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1])
            }
        else:
            # For multiclass, return the raw confusion matrix
            cm_dict = {"matrix": cm.tolist(), "classes": np.unique(y_test).tolist()}
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        return {
            "metrics": metrics,
            "confusion_matrix": cm_dict,
            "execution_time_seconds": execution_time,
            "is_multiclass": is_multiclass
        }
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def load_test_data():
    """Load test data for model evaluation"""
    try:
        if not TEST_DATA_PATH.exists():
            raise FileNotFoundError(f"Test data file not found at {TEST_DATA_PATH}")
        
        df = pd.read_csv(TEST_DATA_PATH)
        
        # Filter out non-feature columns (ID columns and similar)
        non_feature_columns = ['encounter_id', 'patient_nbr']
        feature_columns = [col for col in df.columns if col not in non_feature_columns]
        
        if 'readmitted' not in feature_columns:
            raise ValueError("Target column 'readmitted' not found in test data")
        
        # Log the columns being used
        logger.info(f"Total columns in test data: {len(df.columns)}")
        logger.info(f"Using {len(feature_columns)} columns for features/target")
        
        if set(non_feature_columns).intersection(set(df.columns)):
            logger.info(f"Excluding non-feature columns: {set(non_feature_columns).intersection(set(df.columns))}")
        
        # Filter to only use feature columns
        df = df[feature_columns]
        
        # Assume the target column is 'readmitted'
        X = df.drop('readmitted', axis=1)
        y = df['readmitted']
        
        return X, y
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise

@app.post("/predict-normalized/{model_name}")
async def predict_normalized(
    model_name: str, 
    request: BatchPredictionRequest = None,
    model_path: Optional[str] = Query(None, description="Path to the model file")
):
    """Make predictions using a specified model with normalized features"""
    try:
        logger.info(f"Received prediction request for model: {model_name}")
        
        # Load the specified model
        model, model_file = load_model(model_name, model_path)
        logger.info(f"Loaded model from {model_file}")
        
        if not request or not request.data:
            raise HTTPException(status_code=400, detail="No data provided for prediction")
        
        # Convert input data to DataFrame
        logger.info(f"Received {len(request.data)} records for prediction")
        input_data = []
        for item in request.data:
            # Convert to dict and exclude any None values
            input_dict = {k: v for k, v in item.dict().items() if v is not None}
            input_data.append(input_dict)
        
        input_df = pd.DataFrame(input_data)
        logger.info(f"Input data shape: {input_df.shape}")
        
        # Store original ID columns for output - these are not used for prediction
        id_columns = {}
        non_feature_columns = ['encounter_id', 'patient_nbr', 'readmitted']
        
        for col in non_feature_columns:
            if col in input_df.columns:
                if col != 'readmitted':  # Handle ID columns
                    id_columns[col] = input_df[col]
                input_df = input_df.drop(col, axis=1)
        
        # Store the target column if it exists in the input
        y_actual = None
        if 'readmitted' in input_data[0]:
            y_actual = [item.get('readmitted') for item in input_data]
            y_actual = [y for y in y_actual if y is not None]
            if y_actual and len(y_actual) == len(input_data):
                y_actual = np.array(y_actual)
        
        # Check if the features match what the model expects
        if hasattr(model, 'feature_names_in_'):
            logger.info(f"Model was trained with features: {model.feature_names_in_}")
            logger.info(f"Input features: {list(input_df.columns)}")
            
            # Ensure only the features the model was trained on are used
            missing_features = set(model.feature_names_in_) - set(input_df.columns)
            extra_features = set(input_df.columns) - set(model.feature_names_in_)
            
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            if extra_features:
                logger.warning(f"Extra features found, will be dropped: {extra_features}")
                # Keep only the features the model was trained on
                input_df = input_df[model.feature_names_in_]
        
        # Start prediction timer
        start_time = time.time()
        
        # Make predictions using the model
        try:
            y_pred = model.predict(input_df)
            
            # Check if this is a multiclass problem
            unique_classes = np.unique(y_pred)
            is_multiclass = len(unique_classes) > 2
            
            # Get probability predictions if available
            y_pred_proba = None
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_df)
                    # For binary classification, take the positive class probability
                    if proba.shape[1] == 2:
                        y_pred_proba = proba[:, 1]
                    else:
                        # For multiclass, we'll use the max probability
                        y_pred_proba = np.max(proba, axis=1)
                else:
                    y_pred_proba = y_pred
            except Exception as e:
                logger.warning(f"Error getting probabilities: {str(e)}")
                y_pred_proba = y_pred
            
            # Calculate metrics if actual values are provided
            metrics = None
            if y_actual is not None and len(y_actual) == len(y_pred):
                # Determine average parameter for metrics
                average_param = 'weighted' if is_multiclass else 'binary'
                logger.info(f"Using average='{average_param}' for metrics (multiclass={is_multiclass})")
                
                metrics = {
                    "accuracy": float(accuracy_score(y_actual, y_pred)),
                    "precision": float(precision_score(y_actual, y_pred, zero_division=0, average=average_param)),
                    "recall": float(recall_score(y_actual, y_pred, zero_division=0, average=average_param)),
                    "f1": float(f1_score(y_actual, y_pred, zero_division=0, average=average_param))
                }
                
                # Add ROC AUC only for binary classification
                if not is_multiclass and y_pred_proba is not None:
                    metrics["roc_auc"] = float(roc_auc_score(y_actual, y_pred_proba))
            
            # Format and return predictions
            results = []
            for i in range(len(y_pred)):
                result = {}
                
                # Add ID columns back to results
                for id_col, values in id_columns.items():
                    result[id_col] = values.iloc[i]
                    
                # Add prediction and probability
                result["prediction"] = int(y_pred[i])
                result["probability"] = float(y_pred_proba[i]) if y_pred_proba is not None else None
                
                # For multiclass, also include all class probabilities if available
                if is_multiclass and hasattr(model, 'predict_proba'):
                    all_probs = model.predict_proba(input_df[i:i+1])[0]
                    result["class_probabilities"] = {str(cls): float(prob) for cls, prob in zip(model.classes_, all_probs)}
                
                # Add actual value if available
                if y_actual is not None and len(y_actual) == len(y_pred):
                    result["actual"] = int(y_actual[i])
                
                results.append(result)
            
            # Calculate total time
            execution_time = time.time() - start_time
            
            return {
                "model": model_name,
                "model_path": model_file,
                "predictions": results,
                "count": len(results),
                "metrics": metrics,
                "execution_time_seconds": execution_time,
                "is_multiclass": is_multiclass,
                "classes": [int(cls) for cls in unique_classes.tolist()]
            }
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_name}/performance")
async def get_model_performance(model_name: str):
    """Get performance metrics for a specified model using test data"""
    try:
        # Load model
        model, model_path = load_model(model_name)
        
        # Load test data
        X_test, y_test = load_test_data()
        
        # Evaluate model
        performance = evaluate_model(model, X_test, y_test)
        
        return {
            "model": model_name,
            "model_path": model_path,
            "performance": performance
        }
    except Exception as e:
        logger.error(f"Error getting model performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test-model/{model_name}")
async def test_model(
    model_name: str, 
    model_path: Optional[str] = None,
    fastMode: bool = Query(False, description="Use a smaller subset of data for faster testing")
):
    """Test a model using a validation dataset and return performance metrics"""
    try:
        # Load model
        model, model_file = load_model(model_name, model_path)
        logger.info(f"Loaded model from {model_file}")
        
        # Load test data
        try:
            X_test, y_test = load_test_data()
            logger.info(f"Loaded test data: {X_test.shape}, {y_test.shape}")
            
            # If fast mode is enabled, use a smaller subset of data
            if fastMode:
                sample_size = min(1000, len(X_test))
                logger.info(f"Using fast mode with {sample_size} samples")
                # Use a consistent sample for reproducibility
                indices = np.random.RandomState(42).choice(len(X_test), sample_size, replace=False)
                X_test = X_test.iloc[indices]
                y_test = y_test[indices]
            
            # Ensure feature compatibility
            if hasattr(model, 'feature_names_in_'):
                model_features = set(model.feature_names_in_)
                test_features = set(X_test.columns)
                
                # Check for missing features
                missing_features = model_features - test_features
                if missing_features:
                    raise ValueError(f"Test data is missing features required by the model: {missing_features}")
                
                # Check for extra features
                extra_features = test_features - model_features
                if extra_features:
                    logger.warning(f"Test data contains extra features not used by the model: {extra_features}")
                    # Keep only the features used by the model
                    X_test = X_test[model.feature_names_in_]
            
            logger.info(f"Feature compatibility confirmed: {len(X_test.columns)} features")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading test data: {str(e)}")
        
        # Get feature importance if available
        feature_importance = None
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                feature_importance = {name: float(imp) for name, imp in zip(X_test.columns, importance)}
            elif hasattr(model, 'coef_'):
                importance = model.coef_[0]
                feature_importance = {name: float(imp) for name, imp in zip(X_test.columns, importance)}
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
        
        # Evaluate model
        try:
            # Measure execution time for the evaluation
            start_time = time.time()
            performance = evaluate_model(model, X_test, y_test)
            execution_time = time.time() - start_time
            performance['execution_time_seconds'] = execution_time
            
            logger.info(f"Model evaluation completed in {execution_time:.2f} seconds")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error evaluating model: {str(e)}")
        
        # Return results
        response = {
            "model": model_name,
            "model_path": model_file,
            "feature_importance": feature_importance,
            "test_data": {
                "shape": X_test.shape,
                "features": X_test.columns.tolist(),
                "fast_mode": fastMode
            },
            "performance": performance
        }
        
        # Add model type information if available
        try:
            model_type = type(model).__name__
            response["model_type"] = model_type
        except:
            pass
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in test model endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-models")
async def compare_models(
    patient_data: Dict[str, Any],
    model_paths: List[str] = Query(default=None, description="List of model paths to compare")
):
    """Compare predictions from multiple models for the same patient data"""
    try:
        if not model_paths or len(model_paths) == 0:
            # If no models specified, find all available models
            model_paths = [str(f) for f in MODEL_DIR.glob("*.pkl")]
            
        if not model_paths:
            raise HTTPException(status_code=404, detail="No models found to compare")
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([patient_data])
        
        # Remove ID columns and non-feature columns if present
        id_columns = {}
        non_feature_columns = ['encounter_id', 'patient_nbr', 'readmitted']
        
        for col in non_feature_columns:
            if col in input_df.columns:
                if col != 'readmitted':  # Handle ID columns
                    id_columns[col] = input_df[col].values[0]
                input_df = input_df.drop(col, axis=1)
        
        # Get the first model to check expected features
        first_model_path = model_paths[0]
        first_model_name = Path(first_model_path).stem
        first_model, _ = load_model(model_type=first_model_name, model_path=first_model_path)
        
        # Check if the model has expected feature names
        if hasattr(first_model, 'feature_names_in_'):
            logger.info(f"Model expects features: {first_model.feature_names_in_}")
            
            # Check for missing or extra features
            missing_features = set(first_model.feature_names_in_) - set(input_df.columns)
            extra_features = set(input_df.columns) - set(first_model.feature_names_in_)
            
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            if extra_features:
                logger.warning(f"Extra features found, will be dropped: {extra_features}")
                # Keep only the features the model was trained on
                input_df = input_df[first_model.feature_names_in_]
        
        # Compare models
        results = []
        for model_path in model_paths:
            try:
                # Extract model name from path
                model_name = Path(model_path).stem
                
                # Load model
                model, _ = load_model(model_type=model_name, model_path=model_path)
                
                # Make prediction
                y_pred = model.predict(input_df)[0]
                
                # Get probability if available
                probability = None
                try:
                    probability = model.predict_proba(input_df)[0, 1]
                except:
                    probability = float(y_pred)
                
                # Record result
                model_result = {
                    "model_name": model_name,
                    "model_path": model_path,
                    "prediction": int(y_pred),
                    "probability": float(probability),
                    "model_type": type(model).__name__
                }
                
                results.append(model_result)
            
            except Exception as e:
                logger.warning(f"Error with model {model_path}: {str(e)}")
                # Continue with next model
        
        if not results:
            raise HTTPException(status_code=500, detail="No successful predictions from any model")
        
        # Aggregate results
        predictions = [r["prediction"] for r in results]
        probabilities = [r["probability"] for r in results]
        
        return {
            "patient": id_columns,
            "input_features": patient_data,
            "model_results": results,
            "consensus": {
                "prediction": int(np.round(np.mean(predictions))),
                "avg_probability": float(np.mean(probabilities)),
                "std_probability": float(np.std(probabilities)),
                "model_count": len(results)
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add API routes aliases for backward compatibility
@app.get("/api/health")
async def api_health_check():
    """Legacy route for API health check"""
    return await health_check()

@app.get("/api/available-models")
async def api_get_available_models():
    """Legacy route for available models"""
    return await get_available_models()

@app.get("/api/visualization-data")
async def api_get_visualization_data(
    limit: int = Query(1000, description="Number of records to return", ge=1, le=10000),
    include_stats: bool = Query(True, description="Whether to include column statistics")
):
    """Legacy route for visualization data"""
    return await get_visualization_data(limit=limit, include_stats=include_stats)

@app.post("/api/predict-normalized/{model_name}")
async def api_predict_normalized(
    model_name: str, 
    request: BatchPredictionRequest = None,
    model_path: Optional[str] = Query(None, description="Path to the model file")
):
    """Legacy route for normalized prediction"""
    return await predict_normalized(model_name=model_name, request=request, model_path=model_path)

@app.post("/api/upload-data")
async def api_upload_data(file: UploadFile = File(...)):
    """Legacy route for data upload"""
    return await upload_data(file=file)

@app.get("/api/models/{model_name}/performance")
async def api_get_model_performance(model_name: str):
    """Legacy route for model performance"""
    return await get_model_performance(model_name=model_name)

@app.post("/api/test-model/{model_name}")
async def api_test_model(
    model_name: str, 
    model_path: Optional[str] = None,
    fastMode: bool = Query(False, description="Use a smaller subset of data for faster testing")
):
    """Legacy route for testing models"""
    return await test_model(model_name=model_name, model_path=model_path, fastMode=fastMode)

@app.post("/api/compare-models")
async def api_compare_models(
    patient_data: Dict[str, Any],
    model_paths: List[str] = Query(default=None, description="List of model paths to compare")
):
    """Legacy route for comparing models"""
    return await compare_models(patient_data=patient_data, model_paths=model_paths) 