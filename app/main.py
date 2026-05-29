import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
import shap
import json
import ast
st.set_page_config(page_title="ROI Lens | Enterprise AI Intelligence", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E2E; padding: 20px; border-radius: 10px;
        text-align: center; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 32px; font-weight: bold; color: #4CAF50; }
    .metric-title { font-size: 14px; color: #A0A0B0; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        paths = pd.read_csv('data/processed/paths.csv')
        attr = pd.read_csv('data/processed/attribution_v2.csv')
        opt = pd.read_csv('data/processed/optimized_budget_v2.csv')
        features = pd.read_csv('data/processed/user_features.csv')
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model('data/models/xgb_conversion.json')
        return paths, attr, opt, features, xgb_model
    except Exception as e:
        return None, None, None, None, None

paths, attr, opt, features, xgb_model = load_data()

st.sidebar.title("ROI Lens")
page = st.sidebar.radio("Navigation", ["Executive Summary", "Persona Intelligence", "Bot & Fraud Detection", "Customer Journeys (Sankey)", "Predictive AI (SHAP)", "Counterfactual Simulator", "GenAI Marketing Copilot"])

if paths is None:
    st.error("Enterprise Data not found. Please run the full pipeline.")
    st.stop()

if page == "Executive Summary":
    st.title("Executive Command Center")
    st.markdown("Enterprise AI Marketing Intelligence Platform optimizing for **LTV & Revenue** using Time-Decay Markov Chains and Shapley Game Theory.")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-title">Total Spend</div><div class="metric-value">₹100.0 Cr</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-title">Detected Fraud</div><div class="metric-value" style="color: #E91E63;">12.0%</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-title">Current Revenue</div><div class="metric-value">₹{opt["total_revenue"].sum()/1e6:.1f} M</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-title">Optimized Revenue</div><div class="metric-value" style="color: #2196F3;">₹{opt["optimized_revenue"].sum()/1e6:.1f} M</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("Revenue Optimization Simulation")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.subheader("Budget Shift")
        fig_spend = go.Figure()
        fig_spend.add_trace(go.Bar(x=opt['channel'], y=opt['spend_cr'], name='Current Spend (Cr)', marker_color='#546E7A'))
        fig_spend.add_trace(go.Bar(x=opt['channel'], y=opt['optimized_spend_cr'], name='Optimized Spend (Cr)', marker_color='#4CAF50'))
        fig_spend.update_layout(barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_spend, use_container_width=True)

    with col_b2:
        st.subheader("Revenue Uplift")
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Bar(x=opt['channel'], y=opt['total_revenue'], name='Current Revenue', marker_color='#FF9800'))
        fig_rev.add_trace(go.Bar(x=opt['channel'], y=opt['optimized_revenue'], name='Predicted Revenue', marker_color='#2196F3'))
        fig_rev.update_layout(barmode='group', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_rev, use_container_width=True)

elif page == "Persona Intelligence":
    st.title("Persona-Level Attribution")
    st.markdown("Understand which channels dominate for different demographic segments.")
    
    persona_rev = paths[paths['conversion']==1].groupby('persona')['revenue'].sum().reset_index()
    fig_p = px.pie(persona_rev, values='revenue', names='persona', title='Revenue by Persona', hole=0.4)
    st.plotly_chart(fig_p)

elif page == "Bot & Fraud Detection":
    st.title("ML Fraud Detection (Isolation Forest)")
    st.markdown("Suspicious click patterns detected via clustering and entropy analysis.")
    
    f_bots = features[features['anomaly_score'] == -1]
    st.error(f"Detected {len(f_bots)} high-risk fraudulent sessions.")
    fig_scatter = px.scatter(features, x='avg_time_diff', y='touch_count', color='anomaly_score', 
                             title="Anomaly Clusters", color_continuous_scale='Bluered')
    st.plotly_chart(fig_scatter)

elif page == "Predictive AI (SHAP)":
    st.title("Explainable AI: Conversion Predictor")
    st.markdown("XGBoost model predicting the probability of conversion based on session behavior.")
    
    feat_cols = ['touch_count', 'avg_time_diff', 'min_time_diff', 'channel_entropy']
    X = features[features['anomaly_score'] == 1][feat_cols].sample(500, random_state=42)
    
    st.markdown("### Global Feature Importance")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    
    # Simple bar chart of absolute SHAP values
    shap_df = pd.DataFrame({'feature': feat_cols, 'importance': np.abs(shap_values).mean(axis=0)}).sort_values('importance')
    fig_shap = px.bar(shap_df, x='importance', y='feature', orientation='h', title="SHAP Feature Impact on Conversion")
    st.plotly_chart(fig_shap)

elif page == "Counterfactual Simulator":
    st.title("Strategic Simulator (What-If)")
    st.markdown("Adjust budgets instantly to see how the Ad Fatigue model predicts revenue changes.")
    
    st.sidebar.markdown("### Budget Knobs")
    new_budgets = {}
    for _, row in opt.iterrows():
        new_budgets[row['channel']] = st.sidebar.slider(f"{row['channel']} Spend (Cr)", 0.0, 30.0, float(row['spend_cr']), 0.5)
        
    sim_revs = []
    for _, row in opt.iterrows():
        b = new_budgets[row['channel']]
        A = row['param_A']
        k = row['param_k']
        rev = A * (1 - np.exp(-k * b)) if A > 0 and k > 0 else 0
        sim_revs.append(rev)
        
    total_sim = sum(sim_revs)
    current_total = opt['total_revenue'].sum()
    diff = total_sim - current_total
    
    st.metric("Simulated Total Revenue", f"₹{total_sim/1e6:.1f} M", f"{diff/1e6:.1f} M vs Current")
    
    sim_df = pd.DataFrame({'Channel': opt['channel'], 'Simulated Revenue': sim_revs})
    fig_sim = px.bar(sim_df, x='Channel', y='Simulated Revenue', title="Projected Revenue by Channel")
    st.plotly_chart(fig_sim)

elif page == "Customer Journeys (Sankey)":
    st.title("Customer Journey Flow (Sankey)")
    st.markdown("Visualizing the most frequent sequences of touchpoints leading to conversion.")
    
    try:
        top_conv = pd.read_csv('data/processed/top_converting_paths.csv')
        
        sources, targets, values = [], [], []
        for _, row in top_conv.iterrows():
            path_nodes = row['Path'].split(' > ')
            val = row['Conversions']
            
            # Start -> first touchpoint
            sources.append('Start')
            targets.append(path_nodes[0] + " (1st)")
            values.append(val)
            
            for i in range(len(path_nodes)-1):
                sources.append(path_nodes[i] + f" ({i+1}th)")
                targets.append(path_nodes[i+1] + f" ({i+2}th)")
                values.append(val)
                
            # last touchpoint -> Conversion
            sources.append(path_nodes[-1] + f" ({len(path_nodes)}th)")
            targets.append('Conversion')
            values.append(val)
            
        nodes = list(set(sources + targets))
        # Ensure Start is at the beginning, Conversion at the end conceptually
        node_indices = {node: i for i, node in enumerate(nodes)}
        
        # Color mapping logic
        colors = []
        for n in nodes:
            if n == 'Start': colors.append('#607D8B')
            elif n == 'Conversion': colors.append('#4CAF50')
            else: colors.append('#2196F3')
            
        fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15, thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = nodes, color = colors
            ),
            link = dict(
              source = [node_indices[s] for s in sources],
              target = [node_indices[t] for t in targets],
              value = values
          ))])
          
        fig.update_layout(title_text="Top 5 High-Conversion Pathways", font_size=12, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Run the Sequence Miner first to generate Journey Analytics.")

elif page == "GenAI Marketing Copilot":
    st.title("🤖 GenAI Marketing Copilot")
    st.markdown("Your autonomous AI Chief Marketing Officer. Ask strategic questions about your budget, ROI, and customer journeys.")
    
    api_key = st.sidebar.text_input("Gemini API Key", type="password")
    
    if not api_key:
        st.warning("Please enter your Google Gemini API Key in the sidebar to activate the Copilot.")
    else:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # Build Context String from Data
        try:
            top_paths = pd.read_csv('data/processed/top_converting_paths.csv')
            top_path_str = top_paths.iloc[0]['Path']
        except:
            top_path_str = "Unknown"
            
        context = f"""
        You are the ROI Lens AI Copilot. You are advising the marketing team.
        Here is the current business data:
        Total Budget: ₹100 Crore.
        Detected Bot Traffic: 12%.
        Current Estimated Revenue: ₹{opt['total_revenue'].sum()/1e6:.1f} Million.
        Optimized Estimated Revenue: ₹{opt['optimized_revenue'].sum()/1e6:.1f} Million.
        Top Converting Path: {top_path_str}
        
        Budget Allocation (Current vs Optimized in Crores):
        {opt[['channel', 'spend_cr', 'optimized_spend_cr']].to_string(index=False)}
        """
        
        # Accept user input
        if prompt := st.chat_input("E.g., Which channel should I cut budget from?"):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.spinner("Analyzing..."):
                try:
                    model = genai.GenerativeModel('gemini-pro')
                    full_prompt = f"System Context:\n{context}\n\nUser Question:\n{prompt}"
                    response = model.generate_content(full_prompt)
                    
                    with st.chat_message("assistant"):
                        st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"API Error: {str(e)}")
