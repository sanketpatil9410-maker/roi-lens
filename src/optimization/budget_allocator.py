import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os
import yaml

def load_config():
    with open('config/settings.yaml', 'r') as f:
        return yaml.safe_load(f)

def run_allocator():
    print("Running Revenue-Weighted Budget Allocator...")
    config = load_config()
    attr = pd.read_csv('data/processed/attribution_v2.csv')
    spend = pd.read_csv('data/raw/campaign_spend.csv')
    paths = pd.read_csv('data/processed/paths.csv')
    
    # Calculate total revenue per channel using Time-Decay Heuristic distribution
    # Actually, we can approximate the revenue by distributing each path's revenue 
    # among channels using the same weights.
    
    # Let's map average revenue per conversion per channel.
    channel_revenues = {ch: 0.0 for ch in config['channels']}
    for _, row in paths.iterrows():
        if row['conversion'] == 1:
            rev = row['revenue']
            # Using simple proportional distribution for revenue to approximate
            import ast
            if isinstance(row['channels'], str):
                channels = ast.literal_eval(row['channels'])
            else:
                channels = row['channels']
                
            total_w = len(channels)
            for ch in channels:
                channel_revenues[ch] += rev * (1.0 / total_w)
                
    df = pd.merge(attr, spend, on='channel')
    df['total_revenue'] = df['channel'].map(channel_revenues)
    
    df['spend_cr'] = df['spend_inr'] / 10000000
    total_budget_cr = config['budget']['total_inr'] / 10000000
    
    # Fatigue Model: Revenue = A * (1 - e^(-k * Spend))
    # We estimate A (saturation point) and k (growth rate) heuristically.
    # We know current spend (x0) and current revenue (y0).
    # Let's assume saturation point A = 1.5 * y0 (we can only get 50% more max)
    # Then y0 = 1.5 * y0 * (1 - e^(-k * x0))
    # 1/1.5 = 1 - e^(-k * x0) => e^(-k * x0) = 1 - 0.666 = 0.333
    # -k * x0 = ln(0.333) => k = -ln(0.333) / x0
    
    params = []
    for _, row in df.iterrows():
        y0 = row['total_revenue']
        x0 = row['spend_cr']
        
        if x0 == 0 or y0 == 0:
            A = 0
            k = 0
        else:
            A = 1.5 * y0
            k = -np.log(0.333) / x0
            
        params.append({'A': A, 'k': k})
        
    df['param_A'] = [p['A'] for p in params]
    df['param_k'] = [p['k'] for p in params]
    
    def objective_function(budgets, params):
        total_rev = 0
        for b, p in zip(budgets, params):
            if p['A'] > 0 and p['k'] > 0:
                total_rev += p['A'] * (1 - np.exp(-p['k'] * b))
        return -total_rev
        
    num_channels = len(df)
    initial_guess = df['spend_cr'].values
    
    constraints = ({'type': 'eq', 'fun': lambda b: sum(b) - total_budget_cr})
    bounds = [(0, total_budget_cr) for _ in range(num_channels)]
    
    result = minimize(objective_function, initial_guess, args=(params,), 
                      method='SLSQP', bounds=bounds, constraints=constraints)
                      
    df['optimized_spend_cr'] = result.x
    
    new_revs = []
    for b, p in zip(result.x, params):
        if p['A'] > 0 and p['k'] > 0:
            new_revs.append(p['A'] * (1 - np.exp(-p['k'] * b)))
        else:
            new_revs.append(0)
    df['optimized_revenue'] = new_revs
    
    print("\nOptimization Results (Revenue Focused):")
    print(df[['channel', 'spend_cr', 'optimized_spend_cr', 'total_revenue', 'optimized_revenue']].round(2))
    
    current_total_rev = df['total_revenue'].sum()
    new_total_rev = sum(new_revs)
    
    print(f"\nCurrent Revenue: {current_total_rev:,.2f}")
    print(f"Optimized Revenue: {new_total_rev:,.2f}")
    print(f"Improvement: {((new_total_rev - current_total_rev)/current_total_rev)*100:.2f}%")
    
    df.to_csv('data/processed/optimized_budget_v2.csv', index=False)
    print("Saved optimized revenue budget to data/processed/optimized_budget_v2.csv")

if __name__ == '__main__':
    run_allocator()
