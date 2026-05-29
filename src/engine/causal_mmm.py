import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import yaml
import os

def load_config():
    with open('config/settings.yaml', 'r') as f:
        return yaml.safe_load(f)

def run_causal_mmm():
    print("Running Causal Media Mix Modeling (MMM)...")
    config = load_config()
    df = pd.read_csv('data/raw/daily_marketing_time_series.csv')
    
    channels = config['channels']
    X = np.zeros((len(df), len(channels)))
    
    # 1. Adstock Transformation (Carryover Effect)
    # We estimate a flat 0.5 decay rate for simplicity in this Ridge model
    decay_rate = 0.5
    for i, channel in enumerate(channels):
        spend = df[f'{channel}_spend'].values
        adstock = np.zeros(len(spend))
        adstock[0] = spend[0]
        for t in range(1, len(spend)):
            adstock[t] = spend[t] + adstock[t-1] * decay_rate
        # Diminishing returns log-transformation
        X[:, i] = np.log1p(adstock)
        
    y = df['total_conversions'].values
    
    # 2. Ridge Regression (Causal Extraction)
    # We use Ridge to handle multicollinearity between channels
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    
    coefficients = model.coef_
    
    # 3. Calculate Causal ROI
    # ROI = (Incremental Conversions * Approx Revenue per Conversion) / Total Spend
    # For a log-linear model, incremental contribution can be approximated by coeff * mean(X)
    
    results = []
    for i, channel in enumerate(channels):
        total_spend = df[f'{channel}_spend'].sum()
        # Approximate incremental conversions caused by this channel
        incremental_conv = coefficients[i] * np.sum(X[:, i])
        
        # Assume an average revenue per conversion of 2500 INR
        incremental_revenue = incremental_conv * 2500
        
        causal_roi = incremental_revenue / total_spend if total_spend > 0 else 0
        
        results.append({
            'channel': channel,
            'causal_roi': causal_roi,
            'incremental_revenue': incremental_revenue
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv('data/processed/causal_roi.csv', index=False)
    print("Causal MMM Complete! Results saved to causal_roi.csv")

if __name__ == '__main__':
    run_causal_mmm()
