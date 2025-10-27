#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create sample Titanic dataset for MLOps demonstration
"""

import pandas as pd
import numpy as np
import os

def create_sample_titanic_data():
    """Create a sample Titanic dataset for demonstration"""
    np.random.seed(42)
    
    n_samples = 1000
    
    # Create sample data
    data = {
        'Passengerid': range(1, n_samples + 1),
        '2urvived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),  # 60% died, 40% survived
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
        'Name': [f'Passenger_{i}' for i in range(1, n_samples + 1)],
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
        'Age': np.random.normal(30, 15, n_samples).clip(0, 80),
        'SibSp': np.random.poisson(0.5, n_samples),
        'Parch': np.random.poisson(0.4, n_samples),
        'Ticket': [f'Ticket_{i}' for i in range(1, n_samples + 1)],
        'Fare': np.random.exponential(30, n_samples).clip(0, 500),
        'Cabin': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', np.nan], n_samples, p=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3]),
        'Embarked': np.random.choice(['C', 'Q', 'S', np.nan], n_samples, p=[0.2, 0.1, 0.6, 0.1])
    }
    
    # Add some zero columns for demonstration
    for i in range(19):
        col_name = 'zero' if i == 0 else f'zero.{i}'
        data[col_name] = np.zeros(n_samples)
    
    # Create some correlation between features and survival
    # Higher class passengers more likely to survive
    data['2urvived'] = np.where(
        (data['Pclass'] == 1) & (np.random.random(n_samples) < 0.7), 1,
        np.where(
            (data['Pclass'] == 2) & (np.random.random(n_samples) < 0.5), 1,
            np.where(
                (data['Pclass'] == 3) & (np.random.random(n_samples) < 0.3), 1, 0
            )
        )
    )
    
    # Women more likely to survive
    data['2urvived'] = np.where(
        (data['Sex'] == 'female') & (np.random.random(n_samples) < 0.8), 1,
        data['2urvived']
    )
    
    # Children more likely to survive
    data['2urvived'] = np.where(
        (data['Age'] < 16) & (np.random.random(n_samples) < 0.7), 1,
        data['2urvived']
    )
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    df.loc[missing_indices, 'Age'] = np.nan
    
    missing_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    df.loc[missing_indices, 'Embarked'] = np.nan
    
    return df

def main():
    """Create and save sample dataset"""
    print("Creating sample Titanic dataset...")
    
    # Create sample data
    df = create_sample_titanic_data()
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save dataset
    output_path = 'data/train_and_test2.csv'
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"Sample dataset created with {df.shape[0]} samples and {df.shape[1]} features")
    print(f"Saved to: {output_path}")
    print(f"Survival rate: {df['2urvived'].mean():.2%}")
    print(f"Missing values in Age: {df['Age'].isna().sum()}")
    print(f"Missing values in Embarked: {df['Embarked'].isna().sum()}")

if __name__ == "__main__":
    main()
