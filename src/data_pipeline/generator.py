import pandas as pd
import numpy as np
import uuid
import os
import yaml
from datetime import datetime, timedelta
import random

def load_config():
    with open('config/settings.yaml', 'r') as f:
        return yaml.safe_load(f)

def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def generate_persona(config):
    r = random.random()
    cumulative = 0
    for p_name, p_data in config['personas'].items():
        cumulative += p_data['probability']
        if r <= cumulative:
            return p_name, p_data
    return list(config['personas'].keys())[0], list(config['personas'].values())[0]

def generate_data():
    config = load_config()
    print("Generating v2 mock data for ROI Lens...")
    
    np.random.seed(42)
    random.seed(42)
    
    start_date = datetime.strptime(config['simulation']['start_date'], "%Y-%m-%d")
    end_date = datetime.strptime(config['simulation']['end_date'], "%Y-%m-%d")
    NUM_USERS = config['simulation']['num_users']
    BOT_PERCENTAGE = config['simulation']['bot_percentage']
    CHANNELS = config['channels']
    GEOS = config['geographies']
    
    users = [f"user_{i}" for i in range(NUM_USERS)]
    num_bots = int(NUM_USERS * BOT_PERCENTAGE)
    bots = set(random.sample(users, num_bots))
    
    touchpoints_list = []
    user_profiles_list = []
    
    for user in users:
        is_bot = user in bots
        geo = random.choice(GEOS)
        
        if is_bot:
            # Bot behavior
            num_touches = random.randint(20, 80)
            base_time = random_date(start_date, end_date)
            # Bots don't have personas, assign 'Unknown'
            p_name = 'Unknown'
            
            # Anomaly: very fast, high entropy, repetitive
            for j in range(num_touches):
                ts = base_time + timedelta(seconds=j*0.2 + random.uniform(0, 0.1))
                channel = random.choice(CHANNELS)
                touchpoints_list.append({
                    'interaction_id': str(uuid.uuid4()),
                    'user_id': user,
                    'timestamp': ts,
                    'channel': channel,
                    'interaction_type': 'click'
                })
            
            user_profiles_list.append({
                'user_id': user,
                'persona': p_name,
                'geo': geo,
                'conversion_status': 0,
                'conversion_timestamp': None,
                'revenue_generated': 0.0
            })
            
        else:
            # Human behavior
            p_name, p_data = generate_persona(config)
            num_touches = random.randint(1, 10)
            base_time = random_date(start_date, end_date)
            current_time = base_time
            
            user_channels = []
            for j in range(num_touches):
                # Time decay simulation: gaps get smaller as intent grows
                gap_days = max(0, 3 - j*0.5)
                gap_hours = random.randint(1, 23)
                current_time = current_time + timedelta(days=gap_days, hours=gap_hours)
                
                # Persona affinity
                if random.random() < 0.6:
                    channel = random.choice(p_data['affinity'])
                else:
                    channel = random.choice(CHANNELS)
                    
                user_channels.append(channel)
                touchpoints_list.append({
                    'interaction_id': str(uuid.uuid4()),
                    'user_id': user,
                    'timestamp': current_time,
                    'channel': channel,
                    'interaction_type': 'click'
                })
            
            # Conversion probability
            conv_prob = p_data['base_conversion_rate']
            
            # Synergy logic
            if 'Instagram' in user_channels and 'Google Search' in user_channels:
                conv_prob += 0.10
            
            conversion_status = 1 if random.random() < conv_prob else 0
            
            if conversion_status == 1:
                conv_time = current_time + timedelta(hours=random.randint(1, 24))
                base_revenue = random.uniform(1000, 5000)
                revenue = round(base_revenue * p_data['ltv_multiplier'], 2)
            else:
                conv_time = None
                revenue = 0.0
                
            user_profiles_list.append({
                'user_id': user,
                'persona': p_name,
                'geo': geo,
                'conversion_status': conversion_status,
                'conversion_timestamp': conv_time,
                'revenue_generated': revenue
            })
            
    # Compile
    df_touchpoints = pd.DataFrame(touchpoints_list)
    df_profiles = pd.DataFrame(user_profiles_list)
    
    # Spend data (proportional to channel popularity)
    spend_data = {
        'channel': CHANNELS,
        'spend_inr': [30000000, 25000000, 15000000, 10000000, 8000000, 7000000, 5000000]
    }
    df_spend = pd.DataFrame(spend_data)
    
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    df_touchpoints.to_csv('data/raw/touchpoints.csv', index=False)
    df_profiles.to_csv('data/raw/user_profiles.csv', index=False)
    df_spend.to_csv('data/raw/campaign_spend.csv', index=False)
    
    print("Mock data generated successfully.")
    generate_time_series(config)

def generate_time_series(config, days=365):
    print(f"Generating MMM Time Series Data for {days} days...")
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=days)
    data = {'date': dates}
    
    # Base conversion rate with some seasonality (sine wave)
    seasonality = np.sin(np.linspace(0, 4*np.pi, days)) * 200 + 500
    total_conversions = seasonality
    
    # Generate daily spend per channel and its contribution to conversions
    for channel in config['channels']:
        # Random walk for spend
        spend = np.random.normal(50000, 10000, days)
        spend = np.maximum(spend, 1000) # prevent negative spend
        data[f'{channel}_spend'] = spend
        
        # Simulated True ROI multiplier
        true_roi = np.random.uniform(1.0, 3.5)
        
        # Adstock Effect (Carryover)
        adstock = np.zeros(days)
        adstock[0] = spend[0]
        decay_rate = np.random.uniform(0.3, 0.7)
        for t in range(1, days):
            adstock[t] = spend[t] + adstock[t-1] * decay_rate
            
        diminishing = np.log1p(adstock)
        total_conversions += diminishing * true_roi * 10
        
    data['total_conversions'] = total_conversions.astype(int)
    
    df = pd.DataFrame(data)
    df.to_csv('data/raw/daily_marketing_time_series.csv', index=False)
    print("MMM Time Series Generated!")

if __name__ == '__main__':
    generate_data()
