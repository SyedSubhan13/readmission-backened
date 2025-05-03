"""
Test script for the Readmission Prediction API.
This script tests the API endpoints with sample data.
"""

import requests
import json
import pandas as pd
import time
import os
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"  # Update for Vercel deployment

def test_health_endpoint():
    """Test the health endpoint"""
    print("\n== Testing Health Endpoint ==")
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"API Status: {data.get('status')}")
        print(f"Available Models: {data.get('modelsAvailable', [])}")
        print(f"Model Directory: {data.get('modelDir')}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_available_models():
    """Test the available models endpoint"""
    print("\n== Testing Available Models Endpoint ==")
    response = requests.get(f"{API_BASE_URL}/available-models")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Model Count: {data.get('count', 0)}")
        for model in data.get('models', []):
            print(f"- {model.get('name')} ({model.get('path')})")
        return data.get('models', [])
    else:
        print(f"Error: {response.text}")
        return []

def test_visualization_data(limit=10):
    """Test the visualization data endpoint"""
    print("\n== Testing Visualization Data Endpoint ==")
    params = {"limit": limit, "include_stats": True}
    response = requests.get(f"{API_BASE_URL}/visualization-data", params=params)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        total = data.get('total', 0)
        returned = data.get('returned', 0)
        print(f"Total Records: {total}")
        print(f"Returned Records: {returned}")
        if data.get('data') and len(data['data']) > 0:
            first_record = data['data'][0]
            print(f"Sample Record Keys: {list(first_record.keys())}")
            
            # Check if stats are included
            if data.get('stats'):
                print("Statistics included:")
                if 'numeric' in data['stats']:
                    print(f"- Numeric stats for: {list(data['stats']['numeric'].keys())}")
                if 'counts' in data['stats']:
                    print(f"- Categorical counts for: {list(data['stats']['counts'].keys())}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def test_normalized_prediction(model_name, sample_data=None):
    """Test the normalized prediction endpoint"""
    print(f"\n== Testing Normalized Prediction with {model_name} ==")
    
    # Use sample data or create default data
    if not sample_data:
        sample_data = {
            "data": [
                {
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
            ]
        }
    
    # Make prediction request
    start_time = time.time()
    response = requests.post(
        f"{API_BASE_URL}/predict-normalized/{model_name}",
        json=sample_data
    )
    elapsed = time.time() - start_time
    
    print(f"Status Code: {response.status_code}")
    print(f"Request Time: {elapsed:.2f} seconds")
    
    if response.status_code == 200:
        data = response.json()
        predictions = data.get('predictions', [])
        print(f"Prediction Count: {len(predictions)}")
        
        if predictions:
            for i, pred in enumerate(predictions):
                print(f"Prediction {i+1}:")
                print(f"  - ID: {pred.get('encounter_id', 'N/A')}")
                print(f"  - Prediction: {pred.get('prediction')}")
                print(f"  - Probability: {pred.get('probability'):.4f}")
        
        # Check execution time
        if 'execution_time_seconds' in data:
            print(f"Server Execution Time: {data['execution_time_seconds']:.4f} seconds")
            
        return predictions
    else:
        print(f"Error: {response.text}")
        return []

def test_model_performance(model_name):
    """Test model performance endpoint"""
    print(f"\n== Testing Model Performance for {model_name} ==")
    
    response = requests.get(f"{API_BASE_URL}/models/{model_name}/performance")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        performance = data.get('performance', {})
        
        # Print metrics
        if 'metrics' in performance:
            metrics = performance['metrics']
            print("Performance Metrics:")
            for metric, value in metrics.items():
                print(f"  - {metric}: {value:.4f}")
        
        # Print confusion matrix
        if 'confusion_matrix' in performance:
            cm = performance['confusion_matrix']
            print("Confusion Matrix:")
            print(f"  - True Negatives: {cm.get('tn')}")
            print(f"  - False Positives: {cm.get('fp')}")
            print(f"  - False Negatives: {cm.get('fn')}")
            print(f"  - True Positives: {cm.get('tp')}")
        
        return performance
    else:
        print(f"Error: {response.text}")
        return {}

def test_batch_prediction(model_name, num_samples=5):
    """Test batch prediction with multiple samples"""
    print(f"\n== Testing Batch Prediction with {model_name} ({num_samples} samples) ==")
    
    # Create batch sample data
    samples = []
    for i in range(num_samples):
        # Vary some values slightly for each sample
        samples.append({
            "encounter_id": f"22783{i}2",
            "patient_nbr": f"82221{i}7",
            "time_in_hospital": -1.13 + (i * 0.1),
            "num_lab_procedures": -0.10 + (i * 0.05),
            "num_procedures": -0.78 + (i * 0.02),
            "num_medications": -1.84 + (i * 0.03),
            "number_diagnoses": -3.32 + (i * 0.01),
            "insulin_No": 1.0,
            "diag_3_grouped_other": 1.0,
            "time_in_hospital_x_num_medications": 2.10 + (i * 0.1)
        })
    
    batch_data = {"data": samples}
    
    # Make batch prediction request
    start_time = time.time()
    response = requests.post(
        f"{API_BASE_URL}/predict-normalized/{model_name}",
        json=batch_data
    )
    elapsed = time.time() - start_time
    
    print(f"Status Code: {response.status_code}")
    print(f"Request Time: {elapsed:.2f} seconds")
    
    if response.status_code == 200:
        data = response.json()
        predictions = data.get('predictions', [])
        print(f"Prediction Count: {len(predictions)}")
        
        if predictions and len(predictions) > 0:
            print("Sample of Predictions:")
            for i, pred in enumerate(predictions[:min(3, len(predictions))]):
                print(f"  - Sample {i+1}: Probability {pred.get('probability'):.4f}, Prediction {pred.get('prediction')}")
        
        # Check execution time
        if 'execution_time_seconds' in data:
            print(f"Server Execution Time: {data['execution_time_seconds']:.4f} seconds")
            
        return predictions
    else:
        print(f"Error: {response.text}")
        return []

def test_model_comparison(models=None):
    """Test model comparison with a single patient"""
    print("\n== Testing Model Comparison ==")
    
    # Sample patient data
    patient_data = {
        "time_in_hospital": -0.5,
        "num_lab_procedures": 0.2,
        "num_procedures": -0.3,
        "num_medications": -1.0,
        "number_diagnoses": -2.0,
        "insulin_No": 1.0,
        "diag_3_grouped_other": 1.0,
        "time_in_hospital_x_num_medications": 0.5
    }
    
    # If models are specified, use them for comparison
    params = {}
    if models and len(models) > 0:
        params = {"model_paths": [m.get('path') for m in models]}
    
    # Make comparison request
    response = requests.post(
        f"{API_BASE_URL}/compare-models",
        json=patient_data,
        params=params
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('model_results', [])
        
        print(f"Models Compared: {len(results)}")
        
        if results:
            print("Model Predictions:")
            for res in results:
                print(f"  - {res.get('model_name')}: {res.get('prediction')} ({res.get('probability'):.4f})")
        
        if 'consensus' in data:
            consensus = data['consensus']
            print("\nConsensus:")
            print(f"  - Average Prediction: {consensus.get('prediction')}")
            print(f"  - Average Probability: {consensus.get('avg_probability'):.4f}")
            print(f"  - Standard Deviation: {consensus.get('std_probability'):.4f}")
        
        return results
    else:
        print(f"Error: {response.text}")
        return []

def test_test_model_endpoint(model_name):
    """Test the test-model endpoint for detailed model evaluation"""
    print(f"\n== Testing the test-model Endpoint for {model_name} ==")
    
    response = requests.post(f"{API_BASE_URL}/test-model/{model_name}")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        # Print model information
        print(f"Model Type: {data.get('model_type', 'Unknown')}")
        print(f"Model Path: {data.get('model_path')}")
        
        # Print test data information
        if 'test_data' in data:
            test_data = data['test_data']
            print(f"Test Data Shape: {test_data.get('shape')}")
            print(f"Number of Features: {len(test_data.get('features', []))}")
        
        # Print performance
        if 'performance' in data and 'metrics' in data['performance']:
            metrics = data['performance']['metrics']
            print("\nPerformance Metrics:")
            for metric, value in metrics.items():
                print(f"  - {metric}: {value:.4f}")
        
        # Print feature importance if available
        if 'feature_importance' in data and data['feature_importance']:
            fi = data['feature_importance']
            print("\nTop 5 Important Features:")
            sorted_fi = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)
            for feature, importance in sorted_fi[:5]:
                print(f"  - {feature}: {importance:.4f}")
        
        return data
    else:
        print(f"Error: {response.text}")
        return {}

def main():
    """Run all API tests"""
    print("Starting API Tests")
    print(f"API Base URL: {API_BASE_URL}")
    
    # Test health endpoint
    health_ok = test_health_endpoint()
    if not health_ok:
        print("Health check failed, aborting tests")
        return
    
    # Test available models
    models = test_available_models()
    if not models:
        print("No models available, aborting prediction tests")
    else:
        # Use first model for tests
        model_name = models[0]['name']
        
        # Test prediction endpoints with the model
        test_normalized_prediction(model_name)
        test_batch_prediction(model_name, num_samples=5)
        test_model_performance(model_name)
        test_test_model_endpoint(model_name)
    
    # Test data endpoints
    test_visualization_data(limit=5)
    
    # Test model comparison if multiple models available
    if len(models) > 1:
        test_model_comparison(models)
    else:
        test_model_comparison()
    
    print("\nAPI Tests Completed")

if __name__ == "__main__":
    main() 