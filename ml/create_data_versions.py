#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Versioning Script for MLOps
Creates different versions of the dataset for tracking and comparison
"""

import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime

def create_v1_dataset(input_path, output_path):
    """
    Create v1 dataset - first usable dataset
    Basic preprocessing: remove zero columns, handle missing values
    """
    print("Creating v1 dataset...")
    
    # Read original data
    df = pd.read_csv(input_path, encoding='utf-8')
    print(f"Original dataset shape: {df.shape}")
    
    # Basic preprocessing for v1
    # Remove zero columns
    for i in range(0, 19):
        col_name = 'zero' if i == 0 else f'zero.{i}'
        if col_name in df.columns:
            df.drop([col_name], inplace=True, axis=1)
    
    # Handle missing values with forward fill
    if 'Embarked' in df.columns:
        df['Embarked'].fillna(method='ffill', inplace=True)
    
    # Remove rows with missing target values
    df = df.dropna(subset=['2urvived'])
    
    print(f"V1 dataset shape: {df.shape}")
    
    # Save v1 dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"V1 dataset saved to: {output_path}")
    
    return df

def create_v2_dataset(input_path, output_path):
    """
    Create v2 dataset - improved/cleaned version
    Enhanced preprocessing: outlier removal, feature engineering, balanced sampling
    """
    print("Creating v2 dataset...")
    
    # Read original data
    df = pd.read_csv(input_path, encoding='utf-8')
    print(f"Original dataset shape: {df.shape}")
    
    # Enhanced preprocessing for v2
    # Remove zero columns
    for i in range(0, 19):
        col_name = 'zero' if i == 0 else f'zero.{i}'
        if col_name in df.columns:
            df.drop([col_name], inplace=True, axis=1)
    
    # Handle missing values with more sophisticated methods
    if 'Embarked' in df.columns:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    
    # Remove rows with missing target values
    df = df.dropna(subset=['2urvived'])
    
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
    
    # Balanced sampling to address class imbalance
    if '2urvived' in df.columns:
        survived_counts = df['2urvived'].value_counts()
        min_class_count = min(survived_counts)
        
        # Sample equal number from each class
        balanced_df = df.groupby('2urvived').apply(
            lambda x: x.sample(min(len(x), min_class_count), random_state=42)
        ).reset_index(drop=True)
        
        df = balanced_df
    
    print(f"V2 dataset shape: {df.shape}")
    print(f"Class distribution in v2: {df['2urvived'].value_counts().to_dict()}")
    
    # Save v2 dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"V2 dataset saved to: {output_path}")
    
    return df

def main():
    """Main function to create both dataset versions"""
    # Paths - 使用相对于项目根目录的路径
    original_data = "data/train_and_test2.csv"
    v1_path = "data/train_and_test2_v1.csv"
    v2_path = "data/train_and_test2_v2.csv"
    
    print("=" * 50)
    print("Creating Dataset Versions for MLOps")
    print("=" * 50)
    
    # Create v1 dataset
    df_v1 = create_v1_dataset(original_data, v1_path)
    
    print("\n" + "=" * 30)
    
    # Create v2 dataset
    df_v2 = create_v2_dataset(original_data, v2_path)
    
    print("\n" + "=" * 50)
    print("Dataset Version Summary:")
    print(f"V1 (Basic): {df_v1.shape[0]} samples, {df_v1.shape[1]} features")
    print(f"V2 (Enhanced): {df_v2.shape[0]} samples, {df_v2.shape[1]} features")
    print("=" * 50)

if __name__ == "__main__":
    main()
