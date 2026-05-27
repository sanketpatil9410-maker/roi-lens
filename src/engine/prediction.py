import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import os
import json

def train_predictor():
    print("Training XGBoost Conversion Predictor...")
    df = pd.read_csv('data/processed/user_features.csv')
    
    # We will exclude users who were flagged as anomalies (-1) by Isolation Forest if we saved it.
    if 'anomaly_score' in df.columns:
        df = df[df['anomaly_score'] == 1]
        
    features = ['touch_count', 'avg_time_diff', 'min_time_diff', 'channel_entropy']
    X = df[features]
    y = df['conversion_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Calculate scale_pos_weight since conversions are rare
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0
    
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.1, 
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    
    auc = roc_auc_score(y_test, preds_proba)
    cm = confusion_matrix(y_test, preds)
    
    print(f"Model AUC: {auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Feature Importance
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'feature': features, 'importance': importance}).sort_values('importance', ascending=False)
    
    os.makedirs('data/models', exist_ok=True)
    model.save_model('data/models/xgb_conversion.json')
    feat_imp.to_csv('data/processed/feature_importance.csv', index=False)
    
    print("Saved XGBoost model to data/models/")

if __name__ == '__main__':
    train_predictor()
