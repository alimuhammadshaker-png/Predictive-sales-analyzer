import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from prophet.make_holidays import make_holidays_df
from statsmodels.tsa.statespace.sarimax import SARIMAX
from io import BytesIO
from datetime import datetime, date

# -----------------------------
# Page config & theme
# -----------------------------
st.set_page_config(layout="wide", page_title="💰 Predictive Risk & Opportunity Detector", page_icon="📈")

# Simple theme switcher via CSS
def apply_theme(theme):
    if theme == "Dark":
        st.markdown("""
        <style>
        .stApp { background-color: #0f1117; color: #e5e7eb; }
        .css-18ni7ap, .css-1d391kg, .stMarkdown { color: #e5e7eb !important; }
        .stTextInput > div > div > input, .stSelectbox, .stSlider, .stNumberInput input { color: #e5e7eb !important; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp { background-color: #ffffff; }
        </style>
        """, unsafe_allow_html=True)

# -----------------------------
# Multilingual labels (EN / UR)
# -----------------------------
LANG = st.session_state.get("LANG", "EN")

LABELS = {
    "EN": {
        "title": "💰 Predictive Risk & Opportunity Detector",
        "config": "Configuration",
        "forecast_days": "Forecast Days",
        "holiday_source": "Holiday Source",
        "none": "None",
        "builtin": "Built-in",
        "upload_csv": "Upload CSV",
        "country": "Country",
        "compare_days": "Compare Days",
        "alert_threshold": "Alert Threshold (%)",
        "use_sample": "Use Sample Data",
        "upload_main": "Upload CSV with 'ds' and 'y'",
        "select_reg": "Select Regressors",
        "insights": "Insights",
        "forecast_plot": "Forecast Plot",
        "components": "View Model Components",
        "table": "Forecast Table",
        "download_csv": "Download Forecast CSV",
        "download_xlsx": "Download Forecast Excel",
        "theme": "Theme",
        "language": "Language",
        "live_clock": "Live Clock",
        "scenario": "Scenario Simulation",
        "scenario_help": "Adjust future values of regressors to simulate impact on forecast.",
        "model_compare": "Model Comparison",
        "model_compare_help": "Compare Prophet vs. ARIMA on a hold-out period.",
        "validation": "Data Validation & Summary",
        "anomalies": "Anomaly Detection",
        "custom_holidays": "Custom Holiday Builder",
        "custom_holidays_name": "Holiday name",
        "custom_holidays_dates": "Select holiday dates",
        "build_holidays": "Add custom holidays",
        "status_normal": "Forecast is within normal range.",
        "opportunity": "Opportunity Alert",
        "risk": "Risk Alert",
        "metrics": "Performance Metrics (Hold-out)",
    },
    "UR": {
        "title": "💰 پیش گوئی شدہ رسک اور مواقع ڈٹیکٹر",
        "config": "کنفیگریشن",
        "forecast_days": "فورکاسٹ دن",
        "holiday_source": "چھٹیوں کا ذریعہ",
        "none": "کوئی نہیں",
        "builtin": "بلٹ اِن",
        "upload_csv": "سی ایس وی اپ لوڈ کریں",
        "country": "ملک",
        "compare_days": "تولناتی دن",
        "alert_threshold": "الرٹ تھریش ہولڈ (%)",
        "use_sample": "نمونہ ڈیٹا استعمال کریں",
        "upload_main": "'ds' اور 'y' والی CSV اپ لوڈ کریں",
        "select_reg": "ریگریسرز منتخب کریں",
        "insights": "بصیرت",
        "forecast_plot": "فورکاسٹ پلاٹ",
        "components": "ماڈل کمپوننٹس دیکھیں",
        "table": "فورکاسٹ ٹیبل",
        "download_csv": "فورکاسٹ CSV ڈاؤن لوڈ کریں",
        "download_xlsx": "فورکاسٹ ایکسل ڈاؤن لوڈ کریں",
        "theme": "تھیم",
        "language": "زبان",
        "live_clock": "لائیو گھڑی",
        "scenario": "سیناریو سیمولیشن",
        "scenario_help": "ریگریسرز کی مستقبل کی قدروں کو ایڈجسٹ کریں اور اثر دیکھیں۔",
        "model_compare": "ماڈل موازنہ",
        "model_compare_help": "پرافٹ اور اَریما کا ہولڈ آؤٹ پیریڈ پر تقابل کریں۔",
        "validation": "ڈیٹا ویلیڈیشن اور خلاصہ",
        "anomalies": "انیوملی ڈٹیکشن",
        "custom_holidays": "کسٹم چھٹیاں بنائیں",
        "custom_holidays_name": "چھٹی کا نام",
        "custom_holidays_dates": "چھٹیوں کی تاریخیں منتخب کریں",
        "build_holidays": "کسٹم چھٹیاں شامل کریں",
        "status_normal": "فورکاسٹ نارمل رینج میں ہے۔",
        "opportunity": "موقعہ الرٹ",
        "risk": "رسک الرٹ",
        "metrics": "کارکردگی میٹرکس (ہولڈ آؤٹ)",
    }
}
T = LABELS[LANG]

# Sidebar: theme + language + clock
with st.sidebar:
    st.header(f"⚙️ {T['config']}")
    theme = st.selectbox(f"🎨 {T['theme']}", ["Light", "Dark"], index=0)
    apply_theme(theme)
    LANG = st.selectbox(f"🌐 {T['language']}", ["EN", "UR"], index=0)
    st.session_state["LANG"] = LANG
    T = LABELS[LANG]
    st.caption(f"🕒 {T['live_clock']}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Title
st.markdown(f"<h1 style='text-align: center;'>{T['title']}</h1>", unsafe_allow_html=True)

# -----------------------------
# Sample data
# -----------------------------
@st.cache_data
def generate_sample_data():
    dates = pd.date_range(start="2024-01-01", periods=365)
    np.random.seed(42)
    y = 100 + 0.5 * np.arange(365) + 50 * np.tile([1, 0.5, 0.2, 0.8], 92)[:365] + np.random.normal(0, 10, 365)
    marketing = 10 + np.sin(np.arange(365) / 30) * 5 + np.random.normal(0, 1, 365)
    promo = 5 + np.cos(np.arange(365) / 20) * 3 + np.random.normal(0, 0.8, 365)
    return pd.DataFrame({"ds": dates, "y": y.astype(int), "Marketing_Spend": marketing, "Promo_Intensity": promo})

# -----------------------------
# Forecast with Prophet
# -----------------------------
@st.cache_resource
def generate_forecast_prophet(df, forecast_days, holidays_df=None, regressors=None, reg_future_multipliers=None):
    regressors = regressors or []
    df_fit = df[["ds", "y"] + regressors].copy()
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, holidays=holidays_df)
    for r in regressors:
        model.add_regressor(r)
    model.fit(df_fit)
    future = model.make_future_dataframe(periods=forecast_days)
    for r in regressors:
        base_val = df[r].mean()
        mult = reg_future_multipliers.get(r, 1.0) if reg_future_multipliers else 1.0
        future[r] = base_val * mult
    forecast = model.predict(future)
    forecast = pd.merge(forecast, df[["ds", "y"]], on="ds", how="left")
    return model, forecast

# -----------------------------
# Forecast with ARIMA (SARIMAX)
# -----------------------------
def generate_forecast_arima(df, forecast_days):
    series = df.set_index("ds")["y"].asfreq("D")
    series = series.interpolate()
    # Simple seasonal weekly SARIMAX (adjust if needed)
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.get_prediction(start=series.index[0], end=series.index[-1] + pd.Timedelta(days=forecast_days))
    fc = pred.predicted_mean.reset_index()
    fc.columns = ["ds", "yhat"]
    # Dummy bounds from residual std
    resid_std = np.std(res.resid)
    fc["yhat_lower"] = fc["yhat"] - 1.96 * resid_std
    fc["yhat_upper"] = fc["yhat"] + 1.96 * resid_std
    # Merge actual y
    f = pd.merge(fc, df[["ds", "y"]], on="ds", how="left")
    return f, res

# -----------------------------
# Analysis & insights
# -----------------------------
def analyze_forecast(df_hist, df_forecast, compare_days=30, threshold=0.15):
    alerts = []
    baseline = df_hist["y"].tail(compare_days).mean()
    future = df_forecast[df_forecast["y"].isna()]
    if future.empty:
        alerts.append("⚠️ No forecast data available.")
        return alerts
    next_week = future["yhat"].head(7).mean()
    change = (next_week - baseline) / (baseline if baseline != 0 else 1e-9)
    if change >= threshold:
        alerts.append(f"💰 **{T['opportunity']}**: Next 7 days forecast is {change*100:.1f}% higher than baseline ({baseline:,.0f}).")
    elif change <= -threshold:
        alerts.append(f"📉 **{T['risk']}**: Next 7 days forecast is {abs(change)*100:.1f}% lower than baseline ({baseline:,.0f}).")
    else:
        alerts.append(f"✅ {T['status_normal']}")
    return alerts

def compute_metrics(actual, pred):
    dfm = pd.DataFrame({"actual": actual, "pred": pred}).dropna()
    mae = np.mean(np.abs(dfm["actual"] - dfm["pred"]))
    rmse = np.sqrt(np.mean((dfm["actual"] - dfm["pred"])**2))
    return mae, rmse

def detect_anomalies(df, z_thresh=3.0):
    y = df["y"]
    z = (y - y.mean()) / (y.std() if y.std() != 0 else 1e-9)
    df_a = df.copy()
    df_a["zscore"] = z
    anomalies = df_a[np.abs(df_a["zscore"]) >= z_thresh]
    return anomalies

# -----------------------------
# Plotly forecast chart
# -----------------------------
def plot_forecast_interactive(forecast, title="📈 Forecast"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["y"], mode="markers", name="Actual", marker=dict(color="black")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast", line=dict(color="#2563eb")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper", line=dict(color="#93c5fd"), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower", line=dict(color="#93c5fd"), fill='tonexty', showlegend=False))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Value", template="plotly_white", height=450)
    return fig

# -----------------------------
# Sidebar: core settings
# -----------------------------
with st.sidebar:
    forecast_days = st.slider(f"📅 {T['forecast_days']}", 30, 365, 90)
    holiday_mode = st.radio(f"🎉 {T['holiday_source']}", [T["none"], T["builtin"], T["upload_csv"]])
    holidays_df = None
    if holiday_mode == T["builtin"]:
        country = st.selectbox(f"🌍 {T['country']}", ["US", "UK", "CA", "IN", "JP", "PK"])
        holidays_df = make_holidays_df([2024, 2025, 2026], country)
    elif holiday_mode == T["upload_csv"]:
        file_h = st.file_uploader("📤 Holiday CSV (cols: ds, holiday)", type="csv")
        if file_h:
            df_h = pd.read_csv(file_h)
            if "ds" in df_h.columns and "holiday" in df_h.columns:
                df_h["ds"] = pd.to_datetime(df_h["ds"])
                holidays_df = df_h[["ds", "holiday"]]

# Custom holiday builder
with st.sidebar:
    st.markdown("---")
    st.subheader(f"📅 {T['custom_holidays']}")
    h_name = st.text_input(f"🏷️ {T['custom_holidays_name']}", value="CustomHoliday")
    h_dates = st.date_input(f"🗓️ {T['custom_holidays_dates']}", [])
    if h_dates:
        h_df = pd.DataFrame({"ds": pd.to_datetime(h_dates), "holiday": h_name})
        holidays_df = h_df if holidays_df is None else pd.concat([holidays_df, h_df]).drop_duplicates()

# Compare & threshold
with st.sidebar:
    compare_days = st.slider(f"📊 {T['compare_days']}", 7, 90, 30)
    threshold = st.slider(f"🚨 {T['alert_threshold']}", 5, 50, 15) / 100.0

# Theme + language already above

# -----------------------------
# Data input
# -----------------------------
st.markdown("---")
c1, c2 = st.columns([1, 2])
with c1:
    if st.button(f"📁 {T['use_sample']}"):
        df = generate_sample_data()
        st.session_state["df"] = df
        st.success("✅ Sample data loaded.")
with c2:
    uploaded = st.file_uploader(f"📤 {T['upload_main']}", type="csv")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            df["ds"] = pd.to_datetime(df["ds"])
            st.session_state["df"] = df
            st.success("✅ Data uploaded.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

df = st.session_state.get("df", None)

# -----------------------------
# Validation & summary
# -----------------------------
if df is not None:
    st.markdown("---")
    st.subheader(f"🧪 {T['validation']}")
    cols = list(df.columns)
    missing = df.isna().sum().sum()
    non_numeric_regs = [c for c in df.columns if c not in ["ds", "y"] and not pd.api.types.is_numeric_dtype(df[c])]
    date_gaps = df["ds"].diff().value_counts().to_dict()
    st.write(f"• Columns: {cols}")
    st.write(f"• Rows: {len(df):,}")
    st.write(f"• Missing values: {missing:,}")
    if non_numeric_regs:
        st.warning(f"• Non-numeric regressors detected: {non_numeric_regs}")
    st.write(f"• Date gaps (delta counts): { {str(k): int(v) for k, v in date_gaps.items()} }")

    # Anomaly detection
    st.subheader(f"🧭 {T['anomalies']}")
    anomalies = detect_anomalies(df)
    if len(anomalies) > 0:
        st.warning(f"Found {len(anomalies)} anomalies (|z| ≥ 3). Showing first 10:")
        st.dataframe(anomalies.head(10))
    else:
        st.success("No significant anomalies detected.")

    # Regressors & scenario sliders
    regressors = [c for c in df.columns if c not in ["ds", "y"] and pd.api.types.is_numeric_dtype(df[c])]
    selected_regs = st.multiselect(f"📌 {T['select_reg']}", regressors, default=[r for r in ["Marketing_Spend", "Promo_Intensity"] if r in regressors])

    st.subheader(f"🧪 {T['scenario']}")
    st.caption(T["scenario_help"])
    reg_future_multipliers = {}
    if selected_regs:
        scols = st.columns(len(selected_regs))
        for i, r in enumerate(selected_regs):
            reg_future_multipliers[r] = scols[i].slider(f"{r} future ×", 0.5, 3.0, 1.0, 0.1)

    # Model comparison
    st.markdown("---")
    st.subheader(f"🔀 {T['model_compare']}")
    st.caption(T["model_compare_help"])
    holdout_days = st.slider("Hold-out days for metrics", 14, 90, 30)
    df_sorted = df.sort_values("ds").reset_index(drop=True)
    train_df = df_sorted.iloc[:-holdout_days]
    test_df = df_sorted.iloc[-holdout_days:]

    with st.spinner("🔄 Training Prophet..."):
        p_model, p_fc = generate_forecast_prophet(train_df, forecast_days=holdout_days, holidays_df=holidays_df, regressors=selected_regs, reg_future_multipliers=reg_future_multipliers)
        p_fc_eval = p_fc.tail(holdout_days)
        p_mae, p_rmse = compute_metrics(test_df["y"], p_fc_eval["yhat"])

    with st.spinner("🔄 Training ARIMA..."):
        a_fc, a_model = generate_forecast_arima(train_df, forecast_days=holdout_days)
        a_fc_eval = a_fc.tail(holdout_days)
        a_mae, a_rmse = compute_metrics(test_df["y"].values, a_fc_eval["yhat"].values)

    st.markdown(f"**{T['metrics']}:**")
    mcols = st.columns(2)
    mcols[0].markdown(f"- **Prophet MAE:** {p_mae:,.2f}\n- **Prophet RMSE:** {p_rmse:,.2f}")
    mcols[1].markdown(f"- **ARIMA MAE:** {a_mae:,.2f}\n- **ARIMA RMSE:** {a_rmse:,.2f}")

    # Full forecast (Prophet with entire history)
    with st.spinner("🔮 Running full forecast..."):
        model, forecast = generate_forecast_prophet(df_sorted, forecast_days, holidays_df, selected_regs, reg_future_multipliers)
        alerts = analyze_forecast(df_sorted, forecast, compare_days, threshold)

    st.markdown("---")
    st.subheader(f"🧠 {T['insights']}")
    for a in alerts:
        if "Opportunity" in a or "موقعہ" in a:
            st.success(a)
        elif "Risk" in a or "رسک" in a:
            st.error(a)
        else:
            st.info(a)

    st.markdown("---")
    st.subheader(f"📈 {T['forecast_plot']}")
    st.plotly_chart(plot_forecast_interactive(forecast, title=f"📈 Forecast for next {forecast_days} days"), use_container_width=True)

    # Model components
    with st.expander(f"🧩 {T['components']}"):
        st.pyplot(model.plot_components(forecast))

    # Forecast table
    with st.expander(f"📋 {T['table']}"):
        st.dataframe(forecast[forecast["y"].isna()][["ds", "yhat", "yhat_lower", "yhat_upper"]], use_container_width=True)

    # Downloads: CSV + Excel
    st.markdown("---")
    dcols = st.columns(2)
    csv_data = forecast.to_csv(index=False).encode("utf-8")
    dcols[0].download_button(f"⬇️ {T['download_csv']}", csv_data, file_name="forecast.csv", mime="text/csv")

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        forecast.to_excel(writer, index=False, sheet_name="Forecast")
        pd.DataFrame({
            "Metric": ["Prophet MAE", "Prophet RMSE", "ARIMA MAE", "ARIMA RMSE"],
            "Value": [p_mae, p_rmse, a_mae, a_rmse]
        }).to_excel(writer, index=False, sheet_name="Metrics")
        anomalies.to_excel(writer, index=False, sheet_name="Anomalies")
    dcols[1].download_button(f"⬇️ {T['download_xlsx']}", buffer.getvalue(), file_name="forecast_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("📌 Upload data or use sample to begin.")
