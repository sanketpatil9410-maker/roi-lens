import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
import math
import yaml
import os

def load_config():
    with open('config/settings.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_data():
    return pd.read_csv('data/processed/paths.csv')

def compute_decay_weights(timestamps, conv_time, decay_lambda):
    # Convert string timestamps back to datetime if necessary
    if isinstance(timestamps, str):
        import ast
        timestamps = ast.literal_eval(timestamps)
    
    # parse
    ts_list = pd.to_datetime(timestamps)
    if pd.isna(conv_time):
        return [1.0] * len(ts_list)
        
    conv_dt = pd.to_datetime(conv_time)
    weights = []
    for ts in ts_list:
        diff_days = (conv_dt - ts).total_seconds() / 86400.0
        # exponential decay
        w = np.exp(-decay_lambda * max(0, diff_days))
        weights.append(w)
    return weights

def run_time_decay_attribution():
    print("Running Time-Decay Attribution Models...")
    config = load_config()
    paths = load_data()
    
    decay_lambda = config['attribution']['time_decay_lambda']
    
    def get_conv_time(row):
        if 'conversion_timestamp' in row and pd.notna(row['conversion_timestamp']):
            return row['conversion_timestamp']
        import ast
        ts = ast.literal_eval(row['timestamps']) if isinstance(row['timestamps'], str) else row['timestamps']
        return ts[-1] if row['conversion'] == 1 else None

    paths['weights'] = paths.apply(lambda row: compute_decay_weights(row['timestamps'], get_conv_time(row), decay_lambda), axis=1)
    
    # 1. TIME DECAY HEURISTIC
    time_decay_attr = defaultdict(float)
    for _, row in paths.iterrows():
        if row['conversion'] == 1:
            if isinstance(row['channels'], str):
                import ast
                channels = ast.literal_eval(row['channels'])
            else:
                channels = row['channels']
                
            weights = row['weights']
            total_w = sum(weights)
            for ch, w in zip(channels, weights):
                time_decay_attr[ch] += (w / total_w)
                
    # 2. WEIGHTED MARKOV CHAIN
    transitions = defaultdict(float)
    state_counts = defaultdict(float)
    
    for _, row in paths.iterrows():
        if isinstance(row['channels'], str):
            import ast
            channels = ast.literal_eval(row['channels'])
        else:
            channels = row['channels']
            
        path = ['Start'] + channels
        if row['conversion'] == 1:
            path.append('Conversion')
        else:
            path.append('Null')
            
        weights = [1.0] + row['weights'] + [row['weights'][-1] if row['weights'] else 1.0]
            
        for i in range(len(path) - 1):
            current_state = path[i]
            next_state = path[i+1]
            w = weights[i+1] # Weight of the destination state
            transitions[(current_state, next_state)] += w
            state_counts[current_state] += w
            
    # Calculate transition probabilities
    trans_matrix = {}
    unique_channels = set()
    for (state, next_state), count in transitions.items():
        trans_matrix[(state, next_state)] = count / state_counts[state]
        if state not in ['Start', 'Conversion', 'Null']:
            unique_channels.add(state)
            
    def simulate_conversion(matrix, removed_channel=None, iterations=10000):
        conversions = 0
        for _ in range(iterations):
            current = 'Start'
            while current not in ['Conversion', 'Null']:
                next_states = [s[1] for s in matrix.keys() if s[0] == current]
                probs = [matrix[(current, ns)] for ns in next_states]
                
                if removed_channel and removed_channel in next_states:
                    idx = next_states.index(removed_channel)
                    next_states[idx] = 'Null'
                
                if sum(probs) == 0:
                    current = 'Null'
                    break
                    
                # Normalize probs
                prob_sum = sum(probs)
                probs = [p/prob_sum for p in probs]
                current = np.random.choice(next_states, p=probs)
                
            if current == 'Conversion':
                conversions += 1
        return conversions / iterations

    np.random.seed(42)
    base_conv = simulate_conversion(trans_matrix)
    
    removal_effects = {}
    for channel in unique_channels:
        rem_conv = simulate_conversion(trans_matrix, removed_channel=channel)
        removal_effects[channel] = (base_conv - rem_conv) / base_conv if base_conv > 0 else 0
        
    total_effect = sum(removal_effects.values())
    total_conversions = paths['conversion'].sum()
    
    markov_attr = {}
    for channel, effect in removal_effects.items():
        markov_attr[channel] = (effect / total_effect) * total_conversions if total_effect > 0 else 0

    # 3. COMBINE
    results = pd.DataFrame([dict(time_decay_attr), markov_attr]).T
    results.columns = ['Time_Decay_Heuristic', 'Decay_Weighted_Markov']
    results = results.fillna(0).round(2)
    
    os.makedirs('data/processed', exist_ok=True)
    results.to_csv('data/processed/attribution_v2.csv', index_label='channel')
    print("V2 Attribution Models Complete. Saved to data/processed/attribution_v2.csv")

if __name__ == '__main__':
    run_time_decay_attribution()
