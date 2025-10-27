# Data Versioning Documentation

## Overview
This document tracks the different versions of our dataset used for machine learning model training and evaluation.

## Data Source
- **Original Dataset**: `data/train_and_test2.csv`
- **Source**: Titanic survival prediction dataset
- **Original Size**: 891 samples with 19+ features
- **Target Variable**: `2urvived` (survival status: 0 = died, 1 = survived)

## Dataset Versions

### Version 1 (v1) - Basic Dataset
- **File**: `data/train_and_test2_v1.csv`
- **DVC Hash**: `74cc1e4c4e7e30ec8a49abf227f5099c`
- **Git Commit**: [Link to commit when available]
- **Preprocessing Applied**:
  - Removed all zero columns (zero, zero.1, zero.2, ..., zero.18)
  - Forward fill for missing values in 'Embarked' column
  - Removed rows with missing target values
  - Basic data cleaning

- **Final Shape**: [Will be updated after running]
- **Class Distribution**: [Will be updated after running]

### Version 2 (v2) - Enhanced Dataset
- **File**: `data/train_and_test2_v2.csv`
- **DVC Hash**: [Will be updated after running]
- **Git Commit**: [Link to commit when available]
- **Preprocessing Applied**:
  - All preprocessing from v1
  - **Outlier Removal**: Applied IQR method to remove outliers from numerical features
  - **Feature Engineering**: Created 'Age_Group' categorical feature from 'Age'
  - **Balanced Sampling**: Applied balanced sampling to address class imbalance
  - **Enhanced Missing Value Handling**: Used mode for categorical missing values

- **Final Shape**: [Will be updated after running]
- **Class Distribution**: [Will be updated after running]

## Changes from v1 → v2

1. **Outlier Detection and Removal**
   - Applied Interquartile Range (IQR) method
   - Removed samples where numerical features were beyond 1.5 × IQR from Q1/Q3
   - This helps reduce noise and improve model performance

2. **Feature Engineering**
   - Created age groups: Child (0-12), Teen (13-18), Adult (19-35), Middle (36-60), Senior (60+)
   - This categorical representation may capture non-linear relationships better

3. **Class Balancing**
   - Applied balanced sampling to ensure equal representation of both classes
   - This addresses the class imbalance issue in the original dataset

4. **Improved Missing Value Handling**
   - Used mode (most frequent value) instead of forward fill for categorical variables
   - More statistically sound approach for categorical data

## Data Version Control (DVC) Setup

### Remote Storage
- **DagsHub Repository**: https://dagshub.com/whitecaptain2003/LastPro
- **DVC Remote**: `storage` (configured to DagsHub)
- **Storage Type**: DagsHub-hosted DVC storage

### Commands Used
```bash
# Initialize DVC (already done)
dvc init

# Add remote storage
dvc remote add -d storage https://dagshub.com/whitecaptain2003/LastPro.dvc

# Track data files
dvc add data/train_and_test2_v1.csv
dvc add data/train_and_test2_v2.csv

# Push to remote
dvc push
```

## Data Lineage

```
Original Dataset (train_and_test2.csv)
├── V1 Processing (Basic cleaning)
│   ├── Remove zero columns
│   ├── Forward fill missing values
│   └── Remove missing targets
│   └── → train_and_test2_v1.csv
│
└── V2 Processing (Enhanced)
    ├── All V1 steps
    ├── Outlier removal (IQR)
    ├── Feature engineering (Age groups)
    ├── Class balancing
    └── → train_and_test2_v2.csv
```

## Usage

### Creating Data Versions
```bash
cd ml
python create_data_versions.py
```

### Training with Specific Data Version
```bash
cd ml
python train_with_mlops.py
```

### Pulling Data from DVC
```bash
# Pull specific version
dvc pull data/train_and_test2_v1.csv.dvc
dvc pull data/train_and_test2_v2.csv.dvc
```

## Data Quality Metrics

| Version | Samples | Features | Missing Values | Class Balance | Outliers Removed |
|---------|---------|----------|----------------|---------------|------------------|
| Original| 891     | 19+      | Yes            | Imbalanced    | No              |
| V1      | [TBD]   | [TBD]    | No             | Imbalanced    | No              |
| V2      | [TBD]   | [TBD]    | No             | Balanced      | Yes             |

## Notes
- Raw data is stored in DagsHub, not in Git repository
- Only `.dvc` pointer files are committed to Git
- Data versions are reproducible through DVC
- Each version corresponds to a specific Git commit for full traceability
