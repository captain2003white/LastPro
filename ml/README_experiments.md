# MLflow Experiment Tracking Documentation

## Overview
This document tracks our machine learning experiments using MLflow, including baseline and improved models across different dataset versions.

## Experiment Setup

### MLflow Configuration
- **Tracking URI**: `https://dagshub.com/whitecaptain2003/LastPro.mlflow/` (DAGsHub remote backend)
- **Experiments**:
  - `Baseline_Experiment`: Baseline SVM models
  - `Improved_Experiment`: Hyperparameter-tuned models

### Tracked Information
For each experiment run, we log:
- **Code Version**: Git commit SHA
- **Dataset Version**: DVC dataset version and hash
- **Hyperparameters**: Model parameters and training configuration
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Artifacts**: Trained models, scalers, confusion matrices, classification reports

## Experiments Summary

### Experiment 1: Baseline Model on V1 Data
- **Model Type**: SVM with default parameters
- **Dataset**: V1 (basic preprocessing)
- **Parameters**:
  - C: 1.0
  - gamma: 0.1
  - kernel: rbf
- **Actual Performance**: F1=0.5087, Accuracy=0.7837

### Experiment 2: Improved Model on V1 Data
- **Model Type**: SVM with hyperparameter tuning
- **Dataset**: V1 (basic preprocessing)
- **Tuning**: Grid search over C, gamma, and kernel parameters
- **Actual Performance**: F1=0.5143, Accuracy=0.7837

### Experiment 3: Baseline Model on V2 Data
- **Model Type**: SVM with default parameters
- **Dataset**: V2 (enhanced preprocessing with outlier removal and feature engineering)
- **Parameters**:
  - C: 1.0
  - gamma: 0.1
  - kernel: rbf
- **Actual Performance**: F1=0.7324, Accuracy=0.7324

### Experiment 4: Improved Model on V2 Data
- **Model Type**: SVM with hyperparameter tuning
- **Dataset**: V2 (enhanced preprocessing)
- **Tuning**: Grid search over C, gamma, and kernel parameters
- **Actual Performance**: F1=0.6667, Accuracy=0.6901

## Production-Worthy Model Selection

### Primary Metric: F1-Score
We optimize for **F1-Score** as our primary metric because:

1. **Class Imbalance Handling**: The Titanic dataset has imbalanced classes (more deaths than survivals)
2. **Business Impact**: Both false positives and false negatives have significant consequences:
   - False Positive (predicted survival, actual death): Wasted resources, false hope
   - False Negative (predicted death, actual survival): Missed opportunity to save lives
3. **Balanced Performance**: F1-Score provides a balanced view of precision and recall
4. **Model Comparison**: F1-Score allows fair comparison across different preprocessing approaches

### Model Selection Criteria
The production-worthy model will be selected based on:
1. **Highest F1-Score** on test set
2. **Stable performance** across different data splits
3. **Reasonable training time** for production deployment
4. **Interpretability** for business stakeholders

### Selected Production Model
Based on our experimental results, the **Baseline Model on V2 Data** is production-worthy because:
- **Highest F1-Score**: 0.7324 (best among all experiments)
- **Balanced Performance**: Good balance between precision and recall
- **Enhanced Data**: V2 data has better preprocessing (outlier removal, feature engineering)
- **Class Balance**: Balanced sampling addresses class imbalance
- **Stable Performance**: Consistent results across different metrics

## Running Experiments

### Command to Run All Experiments
```bash
cd ml
python train_with_mlops.py
```

### Command to View MLflow UI
```bash
# View experiments on DAGsHub
# Open: https://dagshub.com/whitecaptain2003/LastPro.mlflow/
```

### Individual Experiment Commands
```bash
# Run specific experiment
python -c "
from train_with_mlops import train_with_mlflow
train_with_mlflow('data/train_and_test2_v1.csv', 'v1', 'Baseline_Experiment', 'baseline')
"
```

## Experiment Results Tracking

### Metrics Logged
- **accuracy**: Overall classification accuracy
- **precision**: True positives / (True positives + False positives)
- **recall**: True positives / (True positives + False negatives)
- **f1_score**: Harmonic mean of precision and recall
- **test_samples**: Number of test samples
- **train_samples**: Number of training samples

### Artifacts Logged
- **model**: Trained sklearn model (pickle format)
- **scaler**: Fitted StandardScaler for feature normalization
- **confusion_matrix_*.csv**: Confusion matrix as CSV
- **classification_report_*.json**: Detailed classification report

### Parameters Logged
- **git_commit**: Git commit hash for code versioning
- **data_version**: Dataset version (v1 or v2)
- **data_dvc_hash**: DVC hash for data versioning
- **model_type**: Type of model (baseline or improved)
- **param_C**: SVM C parameter
- **param_gamma**: SVM gamma parameter
- **param_kernel**: SVM kernel type

## Model Performance Expectations

| Model | Dataset | Actual F1-Score | Actual Accuracy | Notes |
|-------|---------|-----------------|-----------------|-------|
| Baseline | V1 | 0.5087 | 0.7837 | Basic preprocessing |
| Improved | V1 | 0.5143 | 0.7837 | Hyperparameter tuned |
| Baseline | V2 | **0.7324** | **0.7324** | **Best performance** |
| Improved | V2 | 0.6667 | 0.6901 | Enhanced preprocessing |

## Next Steps

1. **Run Experiments**: Execute all experiments and collect results
2. **Model Selection**: Identify the best performing model based on F1-Score
3. **Model Validation**: Perform cross-validation on the selected model
4. **Production Deployment**: Deploy the selected model using the existing Docker setup
5. **Monitoring**: Set up model performance monitoring in production

## Troubleshooting

### Common Issues
1. **MLflow UI not accessible**: Ensure port 5000 is available
2. **DVC data not found**: Run `dvc pull` to download data from remote
3. **Memory issues**: Reduce dataset size or use data sampling
4. **Permission errors**: Ensure write permissions for `/tmp/mlruns`

### Debug Commands
```bash
# Check DVC status
dvc status

# Check data files
ls -la data/

# View experiments on DAGsHub
# Open: https://dagshub.com/whitecaptain2003/LastPro.mlflow/
```
