import os
import re
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

AI_GRADER_PROMPT_TEMPLATE = """SYSTEM:
You are a strict academic grader. Return ONLY valid JSON.

USER:
Grade this time-series forecasting Streamlit project OUT OF 80 points using the fixed rubric below.
Be strict: do not award points unless evidence is present in the submitted JSON.
Return ONLY JSON exactly matching the schema.

RUBRIC MAX:
Data & integrity: 20
Feature engineering: 15
Modeling & evaluation: 25
Dashboard quality: 10
Presentation & rigor: 10

STRICT CAPS:
- If the project only uses baseline features/models with no meaningful additions, cap total_80 <= 45.
- If time-based split is missing/unclear, cap Modeling & evaluation <= 12.
- If missing timestamps/outliers/resampling are not discussed or evidenced, cap Data & integrity <= 10.
- If no metrics table is present, cap Modeling & evaluation <= 10.
- If no insights are provided, cap Presentation & rigor <= 5.

Return JSON:
{
  "scores": {
    "Data & integrity": int,
    "Feature engineering": int,
    "Modeling & evaluation": int,
    "Dashboard quality": int,
    "Presentation & rigor": int
  },
  "total_80": int,
  "strengths": [string, ...],
  "weaknesses": [string, ...],
  "actionable_improvements": [string, ...]
}

EVIDENCE JSON:
<insert submission.json contents here>
"""

st.set_page_config(
    page_title="Mini Project B - Professional Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# VISUAL STYLE
# =========================================================
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(180deg, #f7fbff 0%, #ffffff 45%);
    }
    .hero-box {
        padding: 28px;
        border-radius: 22px;
        background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #06b6d4 100%);
        color: white;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.18);
        margin-bottom: 22px;
    }
    .hero-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 17px;
        opacity: 0.95;
    }
    .section-card {
        padding: 18px;
        border-radius: 18px;
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        margin-bottom: 16px;
    }
    .kpi-card {
        padding: 18px;
        border-radius: 18px;
        background: linear-gradient(135deg, #eff6ff 0%, #ecfeff 100%);
        border: 1px solid #bfdbfe;
        text-align: center;
        min-height: 110px;
    }
    .kpi-label {
        font-size: 13px;
        color: #475569;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi-value {
        font-size: 26px;
        color: #0f172a;
        font-weight: 800;
        margin-top: 8px;
    }
    .good-note {
        padding: 14px;
        border-radius: 14px;
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #064e3b;
    }
    .warn-note {
        padding: 14px;
        border-radius: 14px;
        background-color: #fff7ed;
        border: 1px solid #fed7aa;
        color: #7c2d12;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">⚡ Tetuan Power Consumption Forecasting</div>
        <div class="hero-subtitle">
            Professional time-series forecasting dashboard with data integrity checks, feature engineering,
            model comparison, visual insights, and AI grading evidence export.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR INPUTS
# =========================================================
st.sidebar.header("Student Information")
student_name = st.sidebar.text_input("Student Name", "Nasr Al Shabani")
student_id = st.sidebar.text_input("Student ID", "PG12S2540674")
project_title = st.sidebar.text_input("Project Title", "Tetuan City Power Consumption Forecasting")
project_goal = st.sidebar.text_area(
    "Project Goal",
    "Forecast electricity demand from historical weather and power consumption time-series data, then evaluate model performance using time-based testing."
)
deployed_url = st.sidebar.text_input("Deployed App URL")
repo_url = st.sidebar.text_input("GitHub Repo URL")

st.sidebar.header("Forecast Controls")
dataset_path = st.sidebar.text_input("Dataset Path", "data/dataset_sample.csv")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    raw_df = load_data(dataset_path)
except Exception as exc:
    st.error(f"Could not load dataset from {dataset_path}: {exc}")
    st.stop()

# =========================================================
# DATASET PREVIEW AND AUDIT
# =========================================================
st.subheader("1. Dataset Preview and Audit")

col_a, col_b = st.columns([2, 1])
with col_a:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write("First 10 rows")
    st.dataframe(raw_df.head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write("Dataset size")
    st.metric("Rows", f"{len(raw_df):,}")
    st.metric("Columns", f"{len(raw_df.columns):,}")
    st.markdown('</div>', unsafe_allow_html=True)

dtype_df = pd.DataFrame({
    "column": raw_df.columns,
    "dtype": [str(x) for x in raw_df.dtypes],
    "missing_percent": ((raw_df.isna().mean()) * 100).round(2).values
})
st.dataframe(dtype_df, use_container_width=True)

all_columns = list(raw_df.columns)
default_timestamp_idx = all_columns.index("DateTime") if "DateTime" in all_columns else 0

numeric_columns = raw_df.select_dtypes(include=np.number).columns.tolist()
if not numeric_columns:
    st.error("No numeric columns found. A numeric target column is required.")
    st.stop()

timestamp_col = st.selectbox(
    "Select Timestamp Column",
    all_columns,
    index=default_timestamp_idx
)

target_col = st.selectbox(
    "Select Target Column",
    numeric_columns,
    index=numeric_columns.index("Zone 1 Power Consumption") if "Zone 1 Power Consumption" in numeric_columns else 0
)

# Keep original invalid counts for evidence
timestamp_parsed = pd.to_datetime(raw_df[timestamp_col], errors="coerce")
target_numeric = pd.to_numeric(raw_df[target_col], errors="coerce")
invalid_timestamp_count = int(timestamp_parsed.isna().sum())
missing_target_count = int(target_numeric.isna().sum())

df = raw_df.copy()
df[timestamp_col] = timestamp_parsed
df[target_col] = target_numeric
df = df.dropna(subset=[timestamp_col, target_col]).sort_values(timestamp_col).reset_index(drop=True)

if df.empty:
    st.error("No valid rows remain after parsing timestamp and target.")
    st.stop()

time_diffs = df[timestamp_col].diff().dropna()
median_frequency = str(time_diffs.median()) if len(time_diffs) else "Unknown"
duplicate_timestamp_count = int(df[timestamp_col].duplicated().sum())

q1 = df[target_col].quantile(0.25)
q3 = df[target_col].quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outlier_count = int(((df[target_col] < lower_fence) | (df[target_col] > upper_fence)).sum())

st.subheader("2. Time-Series Cleaning and Integrity Evidence")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows after cleaning", f"{len(df):,}")
c2.metric("Invalid timestamps removed", f"{invalid_timestamp_count:,}")
c3.metric("Missing targets removed", f"{missing_target_count:,}")
c4.metric("Duplicate timestamps", f"{duplicate_timestamp_count:,}")

st.markdown(
    f"""
    <div class="good-note">
    <b>Integrity summary:</b> Timestamp values are parsed using <code>pd.to_datetime</code>,
    target values are converted using <code>pd.to_numeric</code>, invalid rows are removed,
    and the data is sorted in ascending time order. The observed median time step is
    <b>{median_frequency}</b>. IQR outlier review flagged <b>{outlier_count:,}</b> possible unusual target values.
    </div>
    """,
    unsafe_allow_html=True
)

resample_option = st.selectbox(
    "Optional Resampling",
    ["None", "H", "D"],
    help="None keeps the original interval. H uses hourly average. D uses daily average."
)

forecast_horizon = st.number_input(
    "Forecast Horizon (steps ahead)",
    min_value=1,
    max_value=168,
    value=1
)

ts_df = df[[timestamp_col, target_col]].copy().set_index(timestamp_col)

if resample_option != "None":
    ts_df = ts_df.resample(resample_option).mean()

ts_df = ts_df.dropna().reset_index()

if len(ts_df) < 60:
    st.error("Not enough observations remain after resampling. Choose a smaller resampling interval.")
    st.stop()

# =========================================================
# FEATURE ENGINEERING
# =========================================================
st.subheader("3. Feature Engineering")

feature_df = ts_df.copy()

# Lag features
for lag in [1, 2, 3, 6, 12, 24, 48]:
    feature_df[f"lag_{lag}"] = feature_df[target_col].shift(lag)

# Rolling features shifted first to avoid leakage
feature_df["rolling_mean_6"] = feature_df[target_col].shift(1).rolling(6).mean()
feature_df["rolling_std_6"] = feature_df[target_col].shift(1).rolling(6).std()
feature_df["rolling_mean_24"] = feature_df[target_col].shift(1).rolling(24).mean()
feature_df["rolling_std_24"] = feature_df[target_col].shift(1).rolling(24).std()

# Calendar features
feature_df["hour"] = feature_df[timestamp_col].dt.hour
feature_df["dayofweek"] = feature_df[timestamp_col].dt.dayofweek
feature_df["weekend"] = (feature_df[timestamp_col].dt.dayofweek >= 5).astype(int)
feature_df["month"] = feature_df[timestamp_col].dt.month

# Cyclical time features
feature_df["hour_sin"] = np.sin(2 * np.pi * feature_df["hour"] / 24)
feature_df["hour_cos"] = np.cos(2 * np.pi * feature_df["hour"] / 24)
feature_df["month_sin"] = np.sin(2 * np.pi * feature_df["month"] / 12)
feature_df["month_cos"] = np.cos(2 * np.pi * feature_df["month"] / 12)

# Forecast target
feature_df["y_target"] = feature_df[target_col].shift(-forecast_horizon)

feature_df = feature_df.dropna().reset_index(drop=True)

feature_cols = [
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "lag_24", "lag_48",
    "rolling_mean_6", "rolling_std_6", "rolling_mean_24", "rolling_std_24",
    "hour", "dayofweek", "weekend", "month",
    "hour_sin", "hour_cos", "month_sin", "month_cos"
]

X = feature_df[feature_cols]
y = feature_df["y_target"]

f1, f2, f3 = st.columns(3)
f1.metric("Feature rows", f"{len(feature_df):,}")
f2.metric("Model features", f"{len(feature_cols)}")
f3.metric("Forecast horizon", f"{forecast_horizon} step(s)")

st.dataframe(feature_df[[timestamp_col, target_col, "y_target"] + feature_cols].head(10), use_container_width=True)

# =========================================================
# VISUAL PREVIEW
# =========================================================
st.subheader("4. Visual Data Story")

plot_df = ts_df.copy()
plot_df = plot_df.set_index(timestamp_col)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(plot_df.index[:1000], plot_df[target_col].iloc[:1000], color="#2563eb", linewidth=1.4)
ax.set_title("Power Consumption Time-Series Preview", fontsize=14, fontweight="bold")
ax.set_xlabel("Time")
ax.set_ylabel(target_col)
ax.grid(alpha=0.25)
st.pyplot(fig)

# =========================================================
# STUDENT ADDITIONS — MODELING
# =========================================================
st.subheader("5. STUDENT ADDITIONS — MODELING")

st.info(
    "Modeling method: A time-based split is used. The first 80% of observations are used for training, "
    "and the final 20% are used for testing. This is appropriate for forecasting because future data "
    "is not used to predict the past."
)

results_df = None
comparison_df = None
best_model_name = None

try:
    if len(X) < 100:
        st.warning("Not enough rows for reliable modeling. Try reducing the forecast horizon or changing resampling.")
    else:
        split_index = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        test_time = feature_df[timestamp_col].iloc[split_index:]

        models = {
            "Naive Persistence": None,
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=80,
                random_state=42,
                max_depth=14,
                min_samples_leaf=3,
                n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                random_state=42,
                n_estimators=120,
                learning_rate=0.05,
                max_depth=3
            )
        }

        results = []
        prediction_store = {}

        for model_name, model in models.items():
            if model_name == "Naive Persistence":
                y_pred = X_test["lag_1"].values
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mape = float(np.mean(np.abs((y_test.values - y_pred) / np.maximum(np.abs(y_test.values), 1))) * 100)
            r2 = r2_score(y_test, y_pred)

            results.append({
                "Model": model_name,
                "MAE": round(float(mae), 3),
                "RMSE": round(float(rmse), 3),
                "MAPE (%)": round(float(mape), 3),
                "R2 Score": round(float(r2), 4)
            })

            prediction_store[model_name] = y_pred

        results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
        best_model_name = results_df.iloc[0]["Model"]

        st.dataframe(results_df, use_container_width=True)
        st.success(f"Best model based on lowest RMSE: {best_model_name}")

        comparison_df = pd.DataFrame({
            "Timestamp": test_time.values,
            "Actual": y_test.values,
            "Predicted": prediction_store[best_model_name]
        })

        st.subheader("Actual vs Predicted Forecast")
        forecast_plot_df = comparison_df.set_index("Timestamp")[["Actual", "Predicted"]].head(500)
        st.line_chart(forecast_plot_df)

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.scatter(comparison_df["Actual"].head(2000), comparison_df["Predicted"].head(2000), alpha=0.35, color="#0891b2")
        min_val = min(comparison_df["Actual"].min(), comparison_df["Predicted"].min())
        max_val = max(comparison_df["Actual"].max(), comparison_df["Predicted"].max())
        ax2.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=2)
        ax2.set_title("Prediction Quality: Actual vs Predicted", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Actual")
        ax2.set_ylabel("Predicted")
        ax2.grid(alpha=0.25)
        st.pyplot(fig2)

except Exception as e:
    st.error(f"Modeling section error: {e}")

# =========================================================
# STUDENT ADDITIONS — DASHBOARD
# =========================================================
st.subheader("6. STUDENT ADDITIONS — DASHBOARD")

try:
    dashboard_df = ts_df.copy()
    dashboard_df[timestamp_col] = pd.to_datetime(dashboard_df[timestamp_col], errors="coerce")
    dashboard_df = dashboard_df.dropna(subset=[timestamp_col, target_col])

    avg_power = dashboard_df[target_col].mean()
    max_power = dashboard_df[target_col].max()
    min_power = dashboard_df[target_col].min()
    std_power = dashboard_df[target_col].std()

    st.markdown("### Executive KPI Summary")
    k1, k2, k3, k4 = st.columns(4)
    kpi_values = [
        ("Average Power", f"{avg_power:,.2f}"),
        ("Maximum Power", f"{max_power:,.2f}"),
        ("Minimum Power", f"{min_power:,.2f}"),
        ("Power Std Dev", f"{std_power:,.2f}")
    ]
    for col, (label, value) in zip([k1, k2, k3, k4], kpi_values):
        col.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Demand Trend")
    trend_df = dashboard_df.set_index(timestamp_col)[target_col]
    st.line_chart(trend_df)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("### Daily Average Demand")
        daily_avg = dashboard_df.set_index(timestamp_col)[target_col].resample("D").mean().dropna()
        st.line_chart(daily_avg)

    with chart_col2:
        st.markdown("### Monthly Average Demand")
        monthly_avg = dashboard_df.set_index(timestamp_col)[target_col].resample("M").mean().dropna()
        st.bar_chart(monthly_avg)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.markdown("### Average Consumption by Hour")
        hourly_pattern = (
            dashboard_df.assign(hour=dashboard_df[timestamp_col].dt.hour)
            .groupby("hour")[target_col]
            .mean()
            .reset_index()
        )
        st.bar_chart(hourly_pattern.set_index("hour"))

    with chart_col4:
        st.markdown("### Weekday vs Weekend")
        weekend_pattern = dashboard_df.copy()
        weekend_pattern["Day Type"] = np.where(
            weekend_pattern[timestamp_col].dt.dayofweek >= 5,
            "Weekend",
            "Weekday"
        )
        day_type_summary = weekend_pattern.groupby("Day Type")[target_col].mean().reset_index()
        st.bar_chart(day_type_summary.set_index("Day Type"))

    st.markdown("### Heatmap Picture: Average Demand by Day and Hour")
    heatmap_df = dashboard_df.copy()
    heatmap_df["hour"] = heatmap_df[timestamp_col].dt.hour
    heatmap_df["day_name"] = heatmap_df[timestamp_col].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    heatmap_table = (
        heatmap_df.pivot_table(
            index="day_name",
            columns="hour",
            values=target_col,
            aggfunc="mean"
        )
        .reindex(day_order)
    )

    fig3, ax3 = plt.subplots(figsize=(12, 4.8))
    image = ax3.imshow(heatmap_table, aspect="auto", cmap="YlGnBu")
    ax3.set_title("Average Power Consumption Heatmap", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("Day of Week")
    ax3.set_xticks(range(24))
    ax3.set_yticks(range(len(day_order)))
    ax3.set_yticklabels(day_order)
    fig3.colorbar(image, ax=ax3, label=target_col)
    st.pyplot(fig3)

    peak_hour = int(hourly_pattern.sort_values(target_col, ascending=False).iloc[0]["hour"])
    peak_hour_value = float(hourly_pattern.sort_values(target_col, ascending=False).iloc[0][target_col])
    weekday_avg = float(day_type_summary.loc[day_type_summary["Day Type"] == "Weekday", target_col].iloc[0])
    weekend_avg = float(day_type_summary.loc[day_type_summary["Day Type"] == "Weekend", target_col].iloc[0])
    day_gap = weekday_avg - weekend_avg

    dashboard_insights = [
        f"The highest average demand occurs around hour {peak_hour} with an average value of {peak_hour_value:,.2f}.",
        f"Weekday average demand is {weekday_avg:,.2f}, while weekend average demand is {weekend_avg:,.2f}.",
        f"The weekday-weekend demand gap is {day_gap:,.2f}, which can support staffing and energy planning decisions.",
        "The daily and monthly views show how demand changes across different time scales.",
        "The heatmap highlights high-demand periods by day of week and hour of day."
    ]

    st.markdown("### Business Insights")
    for insight in dashboard_insights:
        st.markdown(f"- {insight}")

except Exception as e:
    st.error(f"Dashboard section error: {e}")
    dashboard_insights = ["Dashboard insights could not be generated because the dashboard section produced an error."]

# =========================================================
# EXPORT EVIDENCE
# =========================================================
data_integrity_notes = [
    "Timestamp column was parsed with pd.to_datetime using errors='coerce'. Invalid timestamps were removed.",
    "Target column was converted to numeric using pd.to_numeric. Missing target values were removed.",
    "Rows were sorted in ascending timestamp order before feature engineering.",
    "Optional resampling is available using hourly or daily mean aggregation.",
    f"Median observed time step after cleaning is {median_frequency}.",
    f"IQR outlier review flagged {outlier_count} possible unusual target values."
]

modeling_notes = [
    "A time-based train/test split was used: first 80% of observations for training and final 20% for testing.",
    "The split avoids random shuffling and reduces time-series leakage.",
    "Compared Naive Persistence, Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting.",
    "Model performance was evaluated with MAE, RMSE, MAPE, and R2 Score.",
    "The best model is selected by the lowest RMSE."
]

dashboard_evidence = [
    "Styled KPI cards are included for average, maximum, minimum, and standard deviation.",
    "A demand trend line chart is included.",
    "A daily average demand chart is included.",
    "A monthly average demand chart is included.",
    "An hourly consumption pattern chart is included.",
    "A weekday versus weekend comparison chart is included.",
    "A heatmap picture shows average demand by day of week and hour."
]

def safe_results_table():
    if isinstance(results_df, pd.DataFrame):
        return results_df.to_dict(orient="records")
    return []

def build_submission_json():
    payload = {
        "student_name": student_name,
        "student_id": student_id,
        "project_title": project_title,
        "project_goal": project_goal,
        "deployed_url": deployed_url,
        "repo_url": repo_url,

        "timestamp_column": timestamp_col,
        "target_column": target_col,
        "rows_used": int(len(feature_df)),
        "forecast_horizon": int(forecast_horizon),
        "resampling": resample_option,

        "timestamp_min": str(df[timestamp_col].min()),
        "timestamp_max": str(df[timestamp_col].max()),
        "median_frequency": median_frequency,
        "invalid_timestamp_count": invalid_timestamp_count,
        "missing_target_count": missing_target_count,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "outlier_review_method": "IQR fences using Q1 - 1.5*IQR and Q3 + 1.5*IQR",
        "possible_outlier_count": outlier_count,
        "time_sorted": bool(df[timestamp_col].is_monotonic_increasing),
        "data_integrity_evidence": data_integrity_notes,

        "features": list(X.columns),
        "feature_engineering_evidence": [
            "Created multiple lag features: lag_1, lag_2, lag_3, lag_6, lag_12, lag_24, and lag_48.",
            "Created rolling mean and rolling standard deviation features using shifted target values to avoid leakage.",
            "Created calendar features: hour, dayofweek, weekend, and month.",
            "Created cyclical time features: hour_sin, hour_cos, month_sin, and month_cos.",
            "Created y_target using shift(-forecast_horizon)."
        ],

        "has_time_based_split": True,
        "train_test_split_method": "First 80% training observations, final 20% testing observations",
        "modeling_evidence": modeling_notes,
        "has_metrics_table": isinstance(results_df, pd.DataFrame),
        "results_table": safe_results_table(),
        "best_model": None if best_model_name is None else str(best_model_name),

        "dashboard_evidence": dashboard_evidence,
        "dashboard_visual_count": len(dashboard_evidence),
        "insights": dashboard_insights,

        "presentation_evidence": [
            "The app has a styled header, colored KPI cards, dataset audit, integrity summary, feature preview, model results, dashboard charts, business insights, and export files.",
            "The project card and submission JSON summarize the forecasting workflow and evidence."
        ]
    }
    return payload

submission_payload = build_submission_json()
submission_json_str = json.dumps(submission_payload, indent=2)

project_card = f"""
# Mini Project B — Professional Forecasting Dashboard

## Student
- Name: {student_name}
- ID: {student_id}

## Project
- Title: {project_title}
- Goal: {project_goal}
- Deployed app URL: {deployed_url}
- GitHub repo URL: {repo_url}

## Dataset
- Rows used for feature table: {len(feature_df)}
- Timestamp column: {timestamp_col}
- Target column: {target_col}
- Timestamp range: {df[timestamp_col].min()} to {df[timestamp_col].max()}
- Median frequency: {median_frequency}
- Possible outliers identified using IQR: {outlier_count}

## Feature Engineering
- Features: {", ".join(X.columns)}
- Forecast horizon: {forecast_horizon} step(s)
- Resampling: {resample_option}

## Modeling
- Time-based split: first 80% train, final 20% test
- Models compared: Naive Persistence, Linear Regression, Ridge Regression, Random Forest, Gradient Boosting
- Metrics: MAE, RMSE, MAPE, R2
- Best model: {best_model_name}

## Dashboard and Insights
{chr(10).join("- " + x for x in dashboard_insights)}
"""

st.subheader("7. Export Files")

st.download_button(
    "Download submission.json",
    data=submission_json_str,
    file_name="submission.json",
    mime="application/json"
)

st.download_button(
    "Download project_card.md",
    data=project_card,
    file_name="project_card.md",
    mime="text/markdown"
)

with st.expander("Preview submission evidence JSON"):
    st.json(submission_payload)

# =========================================================
# AI GRADER
# =========================================================
st.subheader("8. AI Grader (/80)")

api_key = None
try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    api_key = st.text_input("Enter OpenRouter API Key", type="password")

if st.button("Run AI Grader"):
    if not api_key:
        st.error("API key required.")
    else:
        prompt = AI_GRADER_PROMPT_TEMPLATE.replace(
            "<insert submission.json contents here>",
            submission_json_str
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        body = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            raw_output = result["choices"][0]["message"]["content"]

            parsed = None
            try:
                parsed = json.loads(raw_output)
            except Exception:
                match = re.search(r"\{.*\}", raw_output, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(0))
                    except Exception:
                        parsed = None

            if parsed:
                st.json(parsed)
            else:
                st.text(raw_output)

        except Exception as e:
            st.error(f"AI grading failed: {e}")
