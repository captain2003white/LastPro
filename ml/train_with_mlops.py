#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Training Script with MLOps Integration
Integrates DVC for data versioning and MLflow for experiment tracking
"""

import pandas as pd
import numpy as np
import os
import sys
import subprocess
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def get_git_commit_hash():
    """Get current git commit hash"""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        return result.stdout.strip()
    except:
        return "unknown"

def get_dvc_data_version(data_path):
    """Get DVC version information for data file"""
    try:
        dvc_file = data_path + '.dvc'
        if os.path.exists(dvc_file):
            with open(dvc_file, 'r') as f:
                dvc_info = f.read()
            # Extract hash from DVC file
            for line in dvc_info.split('\n'):
                if 'md5:' in line:
                    return line.split('md5:')[1].strip()
        return "unknown"
    except:
        return "unknown"

def preprocess_data(df, version="v1"):
    """Enhanced data preprocessing function"""
    print(f"Preprocessing data for version: {version}")
    
    # Remove zero columns
    for i in range(0, 19):
        col_name = 'zero' if i == 0 else f'zero.{i}'
        if col_name in df.columns:
            df.drop([col_name], inplace=True, axis=1)
    
    # Handle missing values
    if 'Embarked' in df.columns:
        if version == "v1":
            df['Embarked'].fillna(method='ffill', inplace=True)
        else:  # v2
            df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Remove rows with missing target values
    df = df.dropna(subset=['2urvived'])
    
    # Additional preprocessing for v2
    if version == "v2":
        # Outlier removal for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['Passengerid', '2urvived']]
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Feature engineering
        if 'Age' in df.columns:
            df['Age_Group'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], 
                                    labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])
    
    # Prepare features and target
    X = df.drop(['Passengerid', '2urvived'], axis=1)
    y = df['2urvived']
    
    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def train_baseline_model(X, y, test_size=0.3, random_state=42):
    """Train baseline SVM model"""
    print("Training baseline SVM model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    # Train SVM with default parameters
    model = SVC(C=1.0, gamma=0.1, kernel='rbf', random_state=random_state)
    model.fit(X_train_std, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_std)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    return model, scaler, X_test, y_test, metrics

def train_improved_model(X, y, test_size=0.3, random_state=42):
    """Train improved model with hyperparameter tuning"""
    print("Training improved model with hyperparameter tuning...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    
    # Hyperparameter tuning for SVM
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    
    svm = SVC(random_state=random_state)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_std, y_train)
    
    best_model = grid_search.best_estimator_
    
    # Evaluate
    y_pred = best_model.predict(X_test_std)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    return best_model, scaler, X_test, y_test, metrics, grid_search.best_params_

def train_with_mlflow(data_path, data_version, experiment_name, model_type="baseline"):
    """Train model with MLflow tracking"""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("file:///tmp/mlruns")
    mlflow.set_experiment(experiment_name)
    
    # Get version information
    git_commit = get_git_commit_hash()
    dvc_hash = get_dvc_data_version(data_path)
    
    with mlflow.start_run(run_name=f"{model_type}_{data_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log metadata
        mlflow.log_param("git_commit", git_commit)
        mlflow.log_param("data_version", data_version)
        mlflow.log_param("data_dvc_hash", dvc_hash)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("data_path", data_path)
        
        # Load and preprocess data
        df = pd.read_csv(data_path, encoding='utf-8')
        X, y = preprocess_data(df, version=data_version)
        
        # Train model
        if model_type == "baseline":
            model, scaler, X_test, y_test, metrics = train_baseline_model(X, y)
            best_params = {"C": 1.0, "gamma": 0.1, "kernel": "rbf"}
        else:  # improved
            model, scaler, X_test, y_test, metrics, best_params = train_improved_model(X, y)
        
        # Log hyperparameters
        for param, value in best_params.items():
            mlflow.log_param(f"param_{param}", value)
        
        # Log metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        
        # Log additional metrics
        y_pred = model.predict(scaler.transform(X_test))
        mlflow.log_metric("test_samples", len(y_test))
        mlflow.log_metric("train_samples", len(X) - len(y_test))
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log scaler
        mlflow.sklearn.log_model(scaler, "scaler")
        
        # Create and log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], 
                           columns=['Predicted 0', 'Predicted 1'])
        cm_path = f"/tmp/confusion_matrix_{model_type}_{data_version}.csv"
        cm_df.to_csv(cm_path)
        mlflow.log_artifact(cm_path)
        
        # Log classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = f"/tmp/classification_report_{model_type}_{data_version}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)
        
        print(f"\n{model_type.upper()} Model Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        return model, scaler, metrics

def main():
    """Main function to run experiments"""
    load_dotenv()
    
    print("=" * 60)
    print("MLOps Training Pipeline")
    print("=" * 60)
    
    # Data paths
    v1_data = "../data/train_and_test2_v1.csv"
    v2_data = "../data/train_and_test2_v2.csv"
    
    # Check if data versions exist
    if not os.path.exists(v1_data):
        print("Creating data versions...")
        from create_data_versions import main as create_versions
        create_versions()
    
    # Run experiments
    experiments = [
        (v1_data, "v1", "Baseline_Experiment", "baseline"),
        (v1_data, "v1", "Improved_Experiment", "improved"),
        (v2_data, "v2", "Baseline_Experiment", "baseline"),
        (v2_data, "v2", "Improved_Experiment", "improved")
    ]
    
    results = {}
    
    for data_path, data_version, experiment_name, model_type in experiments:
        if os.path.exists(data_path):
            print(f"\n{'='*40}")
            print(f"Running: {model_type} model on {data_version} data")
            print(f"{'='*40}")
            
            try:
                model, scaler, metrics = train_with_mlflow(
                    data_path, data_version, experiment_name, model_type
                )
                results[f"{model_type}_{data_version}"] = metrics
            except Exception as e:
                print(f"Error training {model_type} on {data_version}: {e}")
        else:
            print(f"Data file not found: {data_path}")
    
    # Summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for experiment, metrics in results.items():
        print(f"{experiment}: F1={metrics['f1_score']:.4f}, Acc={metrics['accuracy']:.4f}")
    
    print(f"\nMLflow UI available at: file:///tmp/mlruns")
    print("Use 'mlflow ui' to view detailed results")

if __name__ == "__main__":
    main()
