import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="EpiVision AI - Epidemic Intelligence System", 
    layout="wide",
    page_icon="🦠"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'results' not in st.session_state:
    st.session_state.results = None

# ============================================
# API Configuration - Load from environment variables
# ============================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "groqapi")
GROQ_API_URL = os.getenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.1-70b-versatile")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

def query_llm(prompt, model=DEFAULT_MODEL):
    """Query Groq LLM API with improved error handling"""
    
    # Check if API key is configured
    if GROQ_API_KEY == "your_groq_api_key_here":
        return """⚠️ **Groq API Key Not Configured**

Please configure your Groq API key to use LLM features:

**How to get an API key:**
1. Go to https://console.groq.com
2. Sign up or log in to your account
3. Navigate to "API Keys" section
4. Click "Create API Key"
5. Copy your key (starts with 'gsk_')
6. Add it to your .env file: GROQ_API_KEY=your_key_here

**Current Status:** Using demo responses. Configure API key for real AI-powered analysis."""
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        elif response.status_code == 401:
            return "❌ **Invalid API Key**\n\nYour Groq API key appears to be invalid. Please check your key in the .env file."
        elif response.status_code == 429:
            return "⚠️ **Rate Limit Exceeded**\n\nToo many requests. Please wait a moment and try again."
        else:
            return f"⚠️ API Error: {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "⏰ **Request Timeout**\n\nThe API request timed out. Please try again later."
    except requests.exceptions.ConnectionError:
        return """❌ **Connection Error**

Unable to connect to Groq API. Please check:
- Your internet connection
- If Groq API is accessible
- Your firewall/VPN settings"""
    except Exception as e:
        return f"❌ **Error**: {str(e)}"

def process_epidemic_data(df):
    """Process epidemic data and generate insights"""
    # Check if required columns exist
    if 'new_cases' in df.columns:
        cases_col = 'new_cases'
    elif 'cases' in df.columns:
        cases_col = 'cases'
    else:
        return None
    
    if cases_col:
        total_cases = df[cases_col].sum()
        avg_daily = df[cases_col].mean()
        peak_cases = df[cases_col].max()
        
        # Calculate growth rate
        if len(df) > 1:
            df['growth_rate'] = df[cases_col].pct_change() * 100
        
        return {
            'total_cases': total_cases,
            'avg_daily': avg_daily,
            'peak_cases': peak_cases,
            'duration': len(df),
            'max_growth': df['growth_rate'].max() if 'growth_rate' in df.columns else 0,
            'cases_col': cases_col
        }
    else:
        return None

def generate_predictions(df, days=30):
    """Generate simple predictions with better handling for small numbers"""
    # Find the cases column
    if 'new_cases' in df.columns:
        cases_col = 'new_cases'
    elif 'cases' in df.columns:
        cases_col = 'cases'
    else:
        return None
    
    # Find date column
    if 'date' in df.columns:
        date_col = 'date'
    else:
        return None
    
    # Get last 7 days average for better prediction
    last_7_days = df[cases_col].tail(7).mean()
    last_cases = df[cases_col].iloc[-1]
    
    # Calculate growth rate from recent data
    if len(df) > 7:
        recent_growth = df[cases_col].tail(7).pct_change().mean()
        avg_growth = recent_growth if not pd.isna(recent_growth) else 0.05
    else:
        avg_growth = df[cases_col].pct_change().mean() if len(df) > 1 else 0.05
    
    # Handle small numbers - use absolute values for better visualization
    avg_daily_change = 0
    if last_cases < 10:
        # For very small numbers, use linear growth instead of exponential
        avg_daily_change = df[cases_col].diff().tail(7).mean()
        if pd.isna(avg_daily_change):
            avg_daily_change = 0.5  # Small increment for tiny numbers
    
    # Convert date strings to datetime if needed
    if isinstance(df[date_col].iloc[0], str):
        last_date = datetime.strptime(df[date_col].iloc[-1], '%Y-%m-%d')
    else:
        last_date = df[date_col].iloc[-1]
    
    future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
    
    # Generate predictions based on data scale
    if last_cases < 10:
        # Linear prediction for small numbers
        predicted_cases = []
        current = last_cases
        for i in range(days):
            if avg_daily_change > 0:
                current = current + avg_daily_change
            else:
                current = current * (1 + max(0.01, avg_growth))
            predicted_cases.append(max(0, current))
    else:
        # Exponential prediction for larger numbers
        predicted_cases = [last_cases * (1 + avg_growth) ** i for i in range(1, days+1)]
    
    # Ensure predictions are not negative
    predicted_cases = [max(0, p) for p in predicted_cases]
    
    return pd.DataFrame({'date': future_dates, 'predicted_cases': predicted_cases})

# Main UI
st.title("🦠 EpiVision AI - Epidemic Spread Intelligence System")
st.markdown("---")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Data Input")
    uploaded_file = st.file_uploader("Upload Excel or CSV file", type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.data = df
            st.success("✅ File loaded successfully!")
            st.write("Preview:", df.head())
            st.write("Columns:", list(df.columns))
        except Exception as e:
            st.error(f"Error: {e}")
    
    st.markdown("---")
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Select Model", 
        ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0
    )
    prediction_days = st.slider("Prediction Days", 7, 90, 30)
    
    # API Key Status
    st.markdown("---")
    st.header("🔑 API Status")
    if GROQ_API_KEY and GROQ_API_KEY not in ["your_groq_api_key_here", "gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"]:
        st.success("✅ Groq API Key Configured")
    else:
        st.warning("⚠️ Groq API Key Not Configured")
        st.info("Add GROQ_API_KEY to .env file or get from: https://console.groq.com")
    
    # Handle region selection safely
    selected_region = "All"
    if st.session_state.data is not None and 'region' in st.session_state.data.columns:
        regions = ["All"] + list(st.session_state.data['region'].unique())
        selected_region = st.selectbox("Select Region", regions)

# Filter data by region if needed
filtered_df = None
if st.session_state.data is not None:
    if selected_region != "All" and 'region' in st.session_state.data.columns:
        filtered_df = st.session_state.data[st.session_state.data['region'] == selected_region].copy()
    else:
        filtered_df = st.session_state.data.copy()

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Input Data", "📈 Model Spread", "🔮 Predictions", "💊 Interventions", "📋 Report"])

with tab1:
    if filtered_df is not None:
        st.header("Input Data Overview")
        
        # Calculate stats
        if 'new_cases' in filtered_df.columns:
            total_cases = filtered_df['new_cases'].sum()
            avg_daily = filtered_df['new_cases'].mean()
            peak_cases = filtered_df['new_cases'].max()
            total_deaths = filtered_df['deaths'].sum() if 'deaths' in filtered_df.columns else 0
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Cases", f"{total_cases:,.0f}")
            with col2:
                st.metric("Avg Daily", f"{avg_daily:.1f}")
            with col3:
                st.metric("Peak Cases", f"{peak_cases:,.0f}")
            with col4:
                st.metric("Total Deaths", f"{total_deaths:,.0f}")
        
        st.subheader("Raw Data")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Show data summary
        st.subheader("Data Summary")
        st.write(filtered_df.describe())
    else:
        st.info("👈 Please upload an Excel or CSV file to begin")

with tab2:
    if filtered_df is not None:
        st.header("Epidemic Spread Model")
        
        if 'new_cases' in filtered_df.columns and 'date' in filtered_df.columns:
            # Ensure date is datetime
            filtered_df['date'] = pd.to_datetime(filtered_df['date'])
            
            # Line chart
            fig = px.line(filtered_df, x='date', y='new_cases',
                         title=f"Epidemic Curve - {selected_region}",
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Multiple regions comparison
            if selected_region == "All" and 'region' in st.session_state.data.columns:
                st.subheader("Regional Comparison")
                region_data = st.session_state.data.groupby(['date', 'region'])['new_cases'].sum().reset_index()
                region_data['date'] = pd.to_datetime(region_data['date'])
                fig_region = px.line(region_data, x='date', y='new_cases', color='region',
                                    title="Epidemic Curve by Region")
                st.plotly_chart(fig_region, use_container_width=True)
            
            # Growth rate analysis
            st.subheader("Growth Rate Analysis")
            filtered_df['growth_rate'] = filtered_df['new_cases'].pct_change() * 100
            fig2 = px.bar(filtered_df, x='date', y='growth_rate',
                         title="Daily Growth Rate (%)",
                         color='growth_rate',
                         color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig2, use_container_width=True)
            
            # R-effective visualization
            if 'r_effective' in filtered_df.columns:
                st.subheader("R-effective (Reproduction Number)")
                fig_r = px.line(filtered_df, x='date', y='r_effective',
                               title="R-effective Over Time",
                               markers=True)
                fig_r.add_hline(y=1, line_dash="dash", line_color="red",
                               annotation_text="Epidemic Threshold")
                st.plotly_chart(fig_r, use_container_width=True)
            
            # LLM Analysis
            if st.button("Generate Spread Analysis"):
                with st.spinner("Analyzing spread patterns..."):
                    stats = process_epidemic_data(filtered_df)
                    if stats:
                        r_effective_val = filtered_df['r_effective'].iloc[-1] if 'r_effective' in filtered_df.columns else 'N/A'
                        prompt = f"""Analyze this epidemic spread data for {selected_region}:
                        Total Cases: {stats['total_cases']}
                        Peak Cases: {stats['peak_cases']}
                        Current R-effective: {r_effective_val}
                        
                        Provide insights on:
                        1) Current spread pattern and severity
                        2) Transmission dynamics
                        3) Risk assessment for next 30 days
                        4) Key factors influencing spread"""
                        
                        response = query_llm(prompt, model_choice)
                        st.write(response)
    else:
        st.warning("Please upload data first")

with tab3:
    if filtered_df is not None:
        st.header("Predictions & Forecasting")
        
        # Check if we have enough data
        if len(filtered_df) < 7:
            st.warning("⚠️ Not enough data points for reliable predictions. Need at least 7 days of data.")
        else:
            predictions = generate_predictions(filtered_df, prediction_days)
            
            if predictions is not None:
                # Ensure dates are datetime
                filtered_df['date'] = pd.to_datetime(filtered_df['date'])
                
                # Check the scale of predictions
                max_prediction = predictions['predicted_cases'].max()
                
                # Combined historical + predicted
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=filtered_df['date'],
                    y=filtered_df['new_cases'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue', width=2),
                    marker=dict(size=4)
                ))
                fig.add_trace(go.Scatter(
                    x=predictions['date'],
                    y=predictions['predicted_cases'],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=6)
                ))
                
                # Adjust y-axis based on data scale
                y_max = max(filtered_df['new_cases'].max(), max_prediction) * 1.2
                
                fig.update_layout(
                    title=f"Epidemic Forecast - {selected_region}",
                    xaxis_title="Date",
                    yaxis_title="Cases",
                    hovermode='x unified',
                    yaxis=dict(
                        title="Cases",
                        rangemode="tozero",
                        range=[0, y_max] if y_max > 0 else [0, 10]
                    ),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    peak_prediction = predictions['predicted_cases'].max()
                    st.metric("Peak Prediction", f"{peak_prediction:.1f}")
                
                with col2:
                    final_prediction = predictions['predicted_cases'].iloc[-1]
                    current_cases = filtered_df['new_cases'].iloc[-1]
                    if current_cases > 0:
                        change = ((final_prediction / current_cases) - 1) * 100
                        st.metric(f"Day {prediction_days} Forecast",
                                 f"{final_prediction:.1f}",
                                 delta=f"{change:.1f}%")
                    else:
                        st.metric(f"Day {prediction_days} Forecast", f"{final_prediction:.1f}")
                
                with col3:
                    total_predicted = predictions['predicted_cases'].sum()
                    st.metric(f"Total Next {prediction_days} Days",
                             f"{total_predicted:.1f}")
                
                # Prediction confidence intervals
                st.subheader("Prediction Range with Confidence Interval")
                
                # Dynamic confidence interval based on prediction magnitude
                if max_prediction < 10:
                    ci_80_lower = 0.6
                    ci_80_upper = 1.4
                    ci_95_lower = 0.4
                    ci_95_upper = 1.6
                elif max_prediction < 100:
                    ci_80_lower = 0.7
                    ci_80_upper = 1.3
                    ci_95_lower = 0.5
                    ci_95_upper = 1.5
                else:
                    ci_80_lower = 0.8
                    ci_80_upper = 1.2
                    ci_95_lower = 0.7
                    ci_95_upper = 1.3
                
                # Calculate confidence intervals
                predictions['lower_bound_80'] = predictions['predicted_cases'] * ci_80_lower
                predictions['upper_bound_80'] = predictions['predicted_cases'] * ci_80_upper
                predictions['lower_bound_95'] = predictions['predicted_cases'] * ci_95_lower
                predictions['upper_bound_95'] = predictions['predicted_cases'] * ci_95_upper
                
                # Ensure no negative bounds
                predictions['lower_bound_80'] = predictions['lower_bound_80'].clip(lower=0)
                predictions['lower_bound_95'] = predictions['lower_bound_95'].clip(lower=0)
                
                fig_ci = go.Figure()
                
                # Add the main prediction line
                fig_ci.add_trace(go.Scatter(
                    x=predictions['date'],
                    y=predictions['predicted_cases'],
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='red', width=3),
                    marker=dict(size=8)
                ))
                
                # Add 95% confidence interval (outer)
                fig_ci.add_trace(go.Scatter(
                    x=predictions['date'].tolist() + predictions['date'].tolist()[::-1],
                    y=predictions['upper_bound_95'].tolist() + predictions['lower_bound_95'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.1)',
                    line=dict(color='rgba(255,0,0,0)'),
                    name='95% Confidence Interval',
                    showlegend=True
                ))
                
                # Add 80% confidence interval
                fig_ci.add_trace(go.Scatter(
                    x=predictions['date'].tolist() + predictions['date'].tolist()[::-1],
                    y=predictions['upper_bound_80'].tolist() + predictions['lower_bound_80'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.3)',
                    line=dict(color='rgba(255,0,0,0)'),
                    name='80% Confidence Interval',
                    showlegend=True
                ))
                
                # Add boundary lines for 80% CI
                fig_ci.add_trace(go.Scatter(
                    x=predictions['date'],
                    y=predictions['upper_bound_80'],
                    mode='lines',
                    name='Upper Bound (80%)',
                    line=dict(color='rgba(255,0,0,0.5)', width=1, dash='dash'),
                    showlegend=True
                ))
                
                fig_ci.add_trace(go.Scatter(
                    x=predictions['date'],
                    y=predictions['lower_bound_80'],
                    mode='lines',
                    name='Lower Bound (80%)',
                    line=dict(color='rgba(255,0,0,0.5)', width=1, dash='dash'),
                    showlegend=True
                ))
                
                # Set y-axis range based on data
                y_max_ci = max(predictions['upper_bound_95'].max(), predictions['predicted_cases'].max()) * 1.2
                
                fig_ci.update_layout(
                    title=f"Prediction with Confidence Intervals (80% and 95%) - {selected_region}",
                    xaxis_title="Date",
                    yaxis_title="Predicted Cases",
                    hovermode='x unified',
                    yaxis=dict(
                        rangemode="tozero",
                        range=[0, y_max_ci] if y_max_ci > 0 else [0, 10]
                    ),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=1.05
                    ),
                    width=800,
                    height=500
                )
                
                st.plotly_chart(fig_ci, use_container_width=True)
                
                # Add explanation with data context
                with st.expander("📊 Understanding the Predictions"):
                    st.markdown(f"""
                    **Current Data Context:**
                    - Latest cases: {filtered_df['new_cases'].iloc[-1]:.1f}
                    - Average cases: {filtered_df['new_cases'].mean():.1f}
                    - Peak cases: {filtered_df['new_cases'].max():.1f}
                    
                    **Prediction Notes:**
                    - **80% Confidence Interval (dark red)**: There's an 80% probability that actual cases will fall within this range
                    - **95% Confidence Interval (light red)**: There's a 95% probability that actual cases will fall within this range
                    - With small numbers ({max_prediction:.1f} predicted peak), predictions have higher relative uncertainty
                    - The wider intervals reflect the higher uncertainty when dealing with small case counts
                    """)
            else:
                st.warning("Unable to generate predictions. Please check if your data has 'date' and case columns.")
    else:
        st.warning("Please upload data first")

with tab4:
    if filtered_df is not None:
        st.header("Intervention Impact Analysis")
        
        # Intervention effectiveness
        if 'active_intervention' in filtered_df.columns:
            intervention_impact = filtered_df.groupby('active_intervention').agg({
                'new_cases': 'mean',
                'r_effective': 'mean',
                'positivity_rate': 'mean' if 'positivity_rate' in filtered_df.columns else 'new_cases'
            }).round(2)
            
            # Bar chart
            fig = px.bar(x=intervention_impact.index,
                        y=intervention_impact['new_cases'],
                        title="Average Cases by Intervention Type",
                        labels={'x': 'Intervention', 'y': 'Avg Daily Cases'},
                        color=intervention_impact['new_cases'],
                        color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
            
            # R-effective by intervention
            if 'r_effective' in intervention_impact.columns:
                fig2 = px.bar(x=intervention_impact.index,
                             y=intervention_impact['r_effective'],
                             title="R-effective by Intervention Type",
                             labels={'x': 'Intervention', 'y': 'R-effective'},
                             color=intervention_impact['r_effective'],
                             color_continuous_scale='RdYlGn_r')
                fig2.add_hline(y=1, line_dash="dash", line_color="red")
                st.plotly_chart(fig2, use_container_width=True)
            
            # Resource allocation pie chart
            st.subheader("Resource Allocation Recommendation")
            interventions = intervention_impact.index.tolist()
            # Normalize resource allocation
            max_cases = intervention_impact['new_cases'].max()
            resource_allocation = [max(5, (cases / max_cases) * 100) for cases in intervention_impact['new_cases']]
            
            fig3 = px.pie(values=resource_allocation,
                         names=interventions,
                         title="Recommended Resource Distribution (%)")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No intervention data available. Add 'active_intervention' column to see intervention analysis.")
        
        # LLM recommendations
        if st.button("Generate Policy Recommendations"):
            with st.spinner("Generating recommendations..."):
                recent_data = filtered_df.tail(10).to_dict()
                prompt = f"""Based on this epidemic data for {selected_region}:
                Recent trends: {recent_data}
                
                Provide detailed policy recommendations including:
                1) Immediate actions needed (next 7 days)
                2) Resource allocation priorities
                3) Long-term strategy (next 3 months)
                4) Specific interventions based on current R-effective"""
                
                response = query_llm(prompt, model_choice)
                st.write(response)
    else:
        st.warning("Please upload data first")

with tab5:
    if filtered_df is not None:
        st.header("Complete Analysis Report")
        
        # Generate report
        stats = process_epidemic_data(filtered_df)
        predictions = generate_predictions(filtered_df, prediction_days)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Summary Statistics")
            if stats:
                st.write(f"**Region:** {selected_region}")
                st.write(f"**Total Cases:** {stats['total_cases']:,.0f}")
                st.write(f"**Average Daily Cases:** {stats['avg_daily']:.1f}")
                st.write(f"**Peak Cases:** {stats['peak_cases']:,.0f}")
                st.write(f"**Analysis Period:** {stats['duration']} days")
                if 'r_effective' in filtered_df.columns:
                    st.write(f"**Current R-effective:** {filtered_df['r_effective'].iloc[-1]:.2f}")
        
        with col2:
            st.subheader("🔮 Forecast Summary")
            if predictions is not None:
                st.write(f"**Predicted Peak:** {predictions['predicted_cases'].max():,.1f}")
                st.write(f"**{prediction_days}-Day Total:** {predictions['predicted_cases'].sum():,.1f}")
                peak_date = predictions.loc[predictions['predicted_cases'].idxmax(), 'date']
                st.write(f"**Peak Day:** {peak_date.strftime('%Y-%m-%d')}")
        
        # Visual summary
        st.subheader("📊 Visual Summary")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            # Cases over time
            if 'new_cases' in filtered_df.columns:
                fig_summary = px.line(filtered_df, x='date', y='new_cases',
                                     title="Cases Over Time")
                st.plotly_chart(fig_summary, use_container_width=True)
        
        with col_b:
            # Deaths vs Cases correlation
            if 'deaths' in filtered_df.columns and 'new_cases' in filtered_df.columns:
                fig_scatter = px.scatter(filtered_df, x='new_cases', y='deaths',
                                        title="Deaths vs Cases Correlation",
                                        trendline="ols")
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Final LLM conclusions
        st.subheader("🤖 AI-Powered Analysis")
        
        if st.button("Generate Final Conclusions"):
            with st.spinner("Generating comprehensive analysis..."):
                if stats:
                    r_effective_val = filtered_df['r_effective'].iloc[-1] if 'r_effective' in filtered_df.columns else 'N/A'
                    prompt = f"""Generate a comprehensive epidemic report for {selected_region} based on:
                    Total Cases: {stats['total_cases']}
                    Peak Cases: {stats['peak_cases']}
                    Current R-effective: {r_effective_val}
                    
                    Provide:
                    1) Executive summary of the situation
                    2) Key findings from the data
                    3) Recommended interventions
                    4) Conclusion and outlook"""
                    
                    response = query_llm(prompt, model_choice)
                    st.markdown("### 📊 Final Conclusions")
                    st.write(response)
        
        # Export option
        if st.button("Generate Report"):
            if stats and predictions is not None:
                report = f"""EPIDEMIC SPREAD INTELLIGENCE REPORT
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                Region: {selected_region}
                
                SUMMARY STATISTICS:
                Total Cases: {stats['total_cases']}
                Average Daily: {stats['avg_daily']}
                Peak Cases: {stats['peak_cases']}
                Duration: {stats['duration']} days
                
                PREDICTIONS:
                Next {prediction_days} Days Total: {predictions['predicted_cases'].sum()}
                Expected Peak: {predictions['predicted_cases'].max()}
                
                This is an automated epidemic analysis report.
                """
                st.download_button("📥 Download Report", report,
                                 f"epidemic_report_{selected_region}_{datetime.now().strftime('%Y%m%d')}.txt")
    else:
        st.warning("Please upload data first")

# Footer
st.markdown("---")
st.markdown("🔬 **EpiVision AI** - Epidemic Spread Intelligence System v2.0 | Powered by Groq LLM")
