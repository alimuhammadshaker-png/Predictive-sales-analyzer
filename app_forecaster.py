import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
import matplotlib.pyplot as plt
import numpy as np
import yaml # To handle configuration
from streamlit_authenticator import Authenticate
from yaml.loader import SafeLoader

# --- 0. CONFIGURATION & AUTHENTICATION SETUP ---

# Load user credentials (NOTE: config.yaml must be in the same directory)
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("Configuration file (config.yaml) not found. Please create it and restart.")
    st.stop() 

# Initialize Authenticator
authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# --- 1. Core Forecasting Function ---

@st.cache_resource
def generate_forecast(df: pd.DataFrame, forecast_periods: int, holidays_df: pd.DataFrame = None, regressor_cols: list = []):
    """
    Trains a Prophet model, optionally including holidays and external regressors.
    """
    df_fit = df[['ds', 'y'] + regressor_cols].copy()
    
    # Initialize Model with Holidays (if provided)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays_df
    )
    
    # Add External Regressors (if provided)
    for col in regressor_cols:
        model.add_regressor(col)
    
    # Fit Model
    model.fit(df_fit)
    
    # Prepare Future Dates & Regressor Values
    future_dates = model.make_future_dataframe(periods=forecast_periods)
    
    # Assume future regressor values are the mean of historical values
    if regressor_cols:
        future_regressors = df[regressor_cols].mean().to_dict()
        for col, mean_val in future_regressors.items():
            future_dates[col] = mean_val
    
    # Predict
    forecast = model.predict(future_dates)
    
    # Merge original 'y' values back for analysis
    forecast = pd.merge(
        forecast, 
        df[['ds', 'y']], 
        on='ds', 
        how='left'
    )
    
    # Create Visualization
    fig = model.plot(forecast)
    plt.title(f"Sales Forecast (Next {forecast_periods} Days)")
    
    return model, forecast, fig

# --- 2. Actionable Alert Analysis Function ---

def analyze_forecast(df_hist: pd.DataFrame, df_forecast: pd.DataFrame, comparison_period_days: int = 30, threshold_percent: float = 0.15):
    """
    Analyzes the forecast vs. recent history to generate Risk/Opportunity alerts.
    """
    alerts = []
    
    recent_history = df_hist['y'].tail(comparison_period_days)
    historical_avg = recent_history.mean()
    
    future_forecast = df_forecast[df_forecast['y'].isna()]
    
    if future_forecast.empty:
        alerts.append("⚠️ **Warning:** Forecast dataframe is empty. Cannot run analysis.")
        return alerts

    next_week_prediction = future_forecast['yhat'].head(7).mean()
    percent_change = (next_week_prediction - historical_avg) / historical_avg
    
    opportunity_threshold = threshold_percent
    risk_threshold = -threshold_percent
    
    if percent_change >= opportunity_threshold:
        alerts.append(
            f"💰 **OPPORTUNITY ALERT** 💰\n"
            f"Forecasted sales for the next 7 days are **{percent_change * 100:.1f}% higher** "
            f"than the last {comparison_period_days} days' average (Baseline: ${historical_avg:,.0f}).\n"
            f"**ACTION:** High confidence of surge. Prepare inventory and consider increasing ad spend."
        )
    elif percent_change <= risk_threshold:
        alerts.append(
            f"📉 **RISK ALERT** 📉\n"
            f"Forecasted sales for the next 7 days are **{abs(percent_change) * 100:.1f}% lower** "
            f"than the last {comparison_period_days} days' average (Baseline: ${historical_avg:,.0f}).\n"
            f"**ACTION:** Potential dip or overstock risk. Review marketing efforts or adjust inventory plans."
        )
    else:
        alerts.append(
            f"✅ **STATUS QUO** ✅\n"
            f"Forecasted sales for the next 7 days are within the standard {threshold_percent * 100:.0f}% deviation. No immediate, drastic action required."
        )

    return alerts

# --- 3. STREAMLIT APP LAYOUT & LOGIC ---

st.set_page_config(layout="wide")
st.title("💰 Predictive Risk & Opportunity Detector")

# --- AUTHENTICATION LOGIN WIDGET ---
# ORIGINAL (Causing Error)
name, authentication_status, username = authenticator.login('Login', 'sidebar')

# --- MAIN APP CONTENT WRAPPED HERE ---

if authentication_status:
    # 1. LOGOUT BUTTON
    with st.sidebar:
        with st.sidebar:
    authenticator.logout('Logout', 'sidebar') 
    # ... other sidebar content ...
        st.write(f'Welcome, *{name}*')
        st.header("Analyzer Settings") 

    # 2. THE ENTIRE APPLICATION LOGIC IS NOW INDENTED HERE
    st.markdown("---")

    @st.cache_data
    def generate_sample_data():
        dates = pd.to_datetime(pd.date_range(start='2024-01-01', periods=365, freq='D'))
        np.random.seed(42)
        y_values = (
            100 + 
            0.5 * np.arange(365) + 
            50 * np.tile([1, 0.5, 0.2, 0.8], 92)[:365] + 
            np.random.normal(0, 10, 365)
        )
        marketing_spend = 10 + np.sin(np.arange(365) / 30) * 5 + np.random.normal(0, 1, 365)
        
        sample_df = pd.DataFrame({
            'ds': dates, 
            'y': y_values.astype(int),
            'Marketing_Spend': marketing_spend.astype(int)
        })
        return sample_df

    # --- File Uploader and Main Logic ---

    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = None

    if st.button("Use Sample Data for Demonstration"):
        st.session_state['data_loaded'] = generate_sample_data()
        st.info("Using 365 days of automatically generated sample data, including 'Marketing_Spend' as a regressor.")

    uploaded_file = st.file_uploader(
        "Upload your Historical Sales Data (CSV format, required columns: 'ds' and 'y', optional regressors)", 
        type="csv"
    )

    if uploaded_file is not None:
        try:
            sales_df_temp = pd.read_csv(uploaded_file)
            if 'ds' not in sales_df_temp.columns or 'y' not in sales_df_temp.columns:
                 st.error("Error: CSV must contain columns named **'ds'** (date) and **'y'** (value).")
                 st.session_state['data_loaded'] = None
            else:
                sales_df_temp['ds'] = pd.to_datetime(sales_df_temp['ds'])
                st.session_state['data_loaded'] = sales_df_temp
                st.success("Data successfully loaded and formatted!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.session_state['data_loaded'] = None

    sales_df = st.session_state['data_loaded']

    if sales_df is not None:
        
        # --- Sidebar Configuration (CONTINUED) ---
        with st.sidebar:
            st.subheader("Model Parameters")
            forecast_days = st.slider("Future Days to Forecast:", 30, 365, 90)
            
            # --- Holiday Integration ---
            st.subheader("Holiday and Event Integration")
            
            holiday_source = st.radio(
                "Select Holiday Source:",
                ('Built-in Country', 'Upload Custom CSV')
            )
            
            holidays_df = None
            
            if holiday_source == 'Built-in Country':
                country_options = ['None', 'US', 'UK', 'CA', 'DE', 'FR', 'IN', 'JP']
                selected_country = st.selectbox("Select Country Holidays:", country_options)
                
                if selected_country != 'None':
                    start_year = pd.to_datetime(sales_df['ds']).dt.year.min()
                    end_year = pd.to_datetime(sales_df['ds']).dt.year.max() + 1
                    
                    holidays_df = make_holidays_df(year_list=list(range(start_year, end_year + 1)), 
                                                  country=selected_country)
                    st.success(f"Loaded {len(holidays_df)} built-in holidays for {selected_country}.")
                    
            elif holiday_source == 'Upload Custom CSV':
                uploaded_holidays = st.file_uploader(
                    "Upload Custom Holidays (CSV with 'ds' and 'holiday' columns)",
                    type="csv"
                )
                if uploaded_holidays is not None:
                    try:
                        custom_holidays = pd.read_csv(uploaded_holidays)
                        if 'ds' in custom_holidays.columns and 'holiday' in custom_holidays.columns:
                            custom_holidays['ds'] = pd.to_datetime(custom_holidays['ds'])
                            holidays_df = custom_holidays[['ds', 'holiday']]
                            st.success(f"Loaded {len(holidays_df)} custom holidays/events.")
                        else:
                            st.error("Custom CSV must contain 'ds' (date) and 'holiday' (name) columns.")
                            holidays_df = None
                    except Exception as e:
                        st.error(f"Error reading custom holiday file: {e}")
                        holidays_df = None

            # --- External Regressor Integration ---
            st.subheader("External Regressors")
            data_cols = [col for col in sales_df.columns if col not in ['ds', 'y']]
            regressor_cols = st.multiselect("Select External Factors:", data_cols, default=['Marketing_Spend'] if 'Marketing_Spend' in data_cols else [])
            
            st.subheader("Alert Sensitivity")
            comp_days = st.slider("Baseline Comparison (Days):", 7, 90, 30)
            threshold = st.slider("Change Threshold (%) for Alert:", 5, 50, 15) / 100.0

        
        # --- Run Model and Analysis ---
        with st.spinner('Running advanced forecasting model and analysis...'):
            model, forecast_df, plot_fig = generate_forecast(
                sales_df, 
                forecast_days, 
                holidays_df,
                regressor_cols
            )
            alerts = analyze_forecast(sales_df, forecast_df, comp_days, threshold)
        
        
        # --- Display Results ---
        
        st.markdown("---")
        st.header("1. Actionable Insights")
        
        for alert in alerts:
            if "OPPORTUNITY" in alert:
                st.markdown(f'<div style="background-color:#d4edda; color:#155724; padding:15px; border-left: 5px solid #155724; border-radius:3px; font-weight: bold;">{alert}</div>', unsafe_allow_html=True)
            elif "RISK" in alert:
                st.markdown(f'<div style="background-color:#f8d7da; color:#721c24; padding:15px; border-left: 5px solid #721c24; border-radius:3px; font-weight: bold;">{alert}</div>', unsafe_allow_html=True)
            else:
                st.info(alert)

        st.markdown("---")
        st.header("2. Forecast Visualization")
        st.pyplot(plot_fig)

        # --- Component Plots ---
        st.header("3. Model Components Explained")
        st.markdown("These plots show the underlying patterns detected by the model (Trend, Seasonality, Holidays/Events).")
        
        component_fig = model.plot_components(forecast_df)
        st.pyplot(component_fig)
        
        st.markdown("---")
        st.header("4. Raw Forecast Data")
        st.caption(f"Showing predictions for the next {forecast_days} days. yhat = prediction, yhat_lower/upper = confidence interval.")
        
        future_data = forecast_df[forecast_df['y'].isna()][['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        st.dataframe(future_data, use_container_width=True)

    else:
        st.info("Upload a CSV file or click 'Use Sample Data' to start your analysis.")
        
# --- Authentication Error Messages ---
elif authentication_status == False:
    st.error('Username/password is incorrect')

elif authentication_status == None:
    st.warning('Please enter your username and password')


