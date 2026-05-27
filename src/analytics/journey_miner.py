import pandas as pd
import numpy as np
from collections import Counter
import os

def run_miner():
    print("Running Customer Journey Sequence Miner...")
    paths = pd.read_csv('data/processed/paths.csv')
    
    # Analyze frequent converting paths
    converters = paths[paths['conversion'] == 1]
    non_converters = paths[paths['conversion'] == 0]
    
    conv_paths = Counter(converters['path'].tolist())
    non_conv_paths = Counter(non_converters['path'].tolist())
    
    top_conv = conv_paths.most_common(5)
    top_failed = non_conv_paths.most_common(5)
    
    print("\nTop 5 Converting Journeys:")
    for p, c in top_conv:
        print(f"[{c} conversions] {p}")
        
    print("\nTop 5 Failed Journeys:")
    for p, c in top_failed:
        print(f"[{c} failures] {p}")
        
    # Save to CSV
    df_conv = pd.DataFrame(top_conv, columns=['Path', 'Conversions'])
    df_fail = pd.DataFrame(top_failed, columns=['Path', 'Failures'])
    
    os.makedirs('data/processed', exist_ok=True)
    df_conv.to_csv('data/processed/top_converting_paths.csv', index=False)
    df_fail.to_csv('data/processed/top_failed_paths.csv', index=False)
    
    print("\nSequence Mining Complete. Saved to data/processed/")

if __name__ == '__main__':
    run_miner()
