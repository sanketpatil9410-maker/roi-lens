import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os

def feature_engineering(touchpoints, profiles):
    print("Engineering features for ML Bot Detection...")
    df = pd.merge(touchpoints, profiles, on='user_id', how='left')
    df = df.sort_values(by=['user_id', 'timestamp'])
    
    # Session features
    user_features = []
    grouped = df.groupby('user_id')
    
    for user_id, group in grouped:
        touch_count = len(group)
        if touch_count > 1:
            time_diffs = group['timestamp'].diff().dt.total_seconds().dropna()
            avg_time_diff = time_diffs.mean()
            min_time_diff = time_diffs.min()
            
            # Entropy of channels (diversity)
            channel_counts = group['channel'].value_counts()
            probs = channel_counts / touch_count
            entropy = -sum(probs * np.log2(probs + 1e-9))
        else:
            avg_time_diff = 86400 # 1 day
            min_time_diff = 86400
            entropy = 0
            
        conversion = group['conversion_status'].iloc[0]
        
        user_features.append({
            'user_id': user_id,
            'touch_count': touch_count,
            'avg_time_diff': avg_time_diff,
            'min_time_diff': min_time_diff,
            'channel_entropy': entropy,
            'conversion_status': conversion
        })
        
    return pd.DataFrame(user_features), df

def detect_bots(features_df, df_merged):
    print("Running Isolation Forest for anomaly detection...")
    
    # We only train on non-converters, assuming converters are humans.
    # Anomaly detection looks for bizarre behavior in the non-converting pool.
    
    X = features_df[['touch_count', 'avg_time_diff', 'min_time_diff', 'channel_entropy']]
    
    # Isolation Forest
    iso = IsolationForest(n_estimators=100, contamination=0.12, random_state=42)
    features_df['anomaly_score'] = iso.fit_predict(X)
    
    # -1 is anomaly, 1 is normal
    bots = features_df[features_df['anomaly_score'] == -1]['user_id'].tolist()
    
    print(f"Detected {len(bots)} ML-classified bots out of {len(features_df)} users.")
    
    clean_users = set(features_df['user_id']) - set(bots)
    clean_df = df_merged[df_merged['user_id'].isin(clean_users)]
    
    return clean_df, bots

def generate_paths(clean_df):
    clean_df = clean_df.sort_values(by=['user_id', 'timestamp'])
    
    # To implement time-decay later, we need to save the timestamps of the path.
    # We will save the path as a list of (channel, timestamp)
    paths = clean_df.groupby('user_id').agg(
        path=('channel', lambda x: ' > '.join(x)),
        channels=('channel', list),
        timestamps=('timestamp', lambda x: [t.isoformat() for t in x]),
        conversion=('conversion_status', 'first'),
        revenue=('revenue_generated', 'first'),
        persona=('persona', 'first'),
        geo=('geo', 'first')
    ).reset_index()
    return paths

def run_pipeline():
    touchpoints = pd.read_csv('data/raw/touchpoints.csv', parse_dates=['timestamp'])
    profiles = pd.read_csv('data/raw/user_profiles.csv', parse_dates=['conversion_timestamp'])
    
    features_df, df_merged = feature_engineering(touchpoints, profiles)
    clean_df, bots = detect_bots(features_df, df_merged)
    
    paths_df = generate_paths(clean_df)
    
    os.makedirs('data/processed', exist_ok=True)
    paths_df.to_csv('data/processed/paths.csv', index=False)
    features_df.to_csv('data/processed/user_features.csv', index=False)
    
    print("Data processing & ML Bot Detection complete.")

if __name__ == '__main__':
    run_pipeline()
