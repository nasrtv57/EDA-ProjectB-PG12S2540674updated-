import os
import re
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

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

st.set_page_config(page_title="Mini Project B - Forecasting", layout="wide")

st.title("Mini Project B — Time-Series Forecasting")

st.sidebar.header("Student Information")
student_name = st.sidebar.text_input("Student Name", "Nasr Al Shaabani")
student_id = st.sidebar.text_input("Student ID", "PG12S2540674")
project_title = st.sidebar.text_input("Project Title", "Tetuan Power Consumption Forecasting")
project_goal = st.sidebar.text_area(
    "Project Goal",
    "Forecast electricity consumption using historical time-series data."
)
deployed_url = st.sidebar.text_input("Deployed App URL")

dataset_path = st.text_input("Dataset Path", "data/dataset_sample.csv")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(dataset_path)

st.subheader("First 10 Rows")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Dataset Audit")

dtype_df = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(x) for x in df.dtypes],
    "missing_percent": ((df.isna().mean()) * 100).round(2).values
})

st.dataframe(dtype_df, use_container_width=True)

all_columns = list(df.columns)
default_timestamp_idx = all_columns.index("DateTime") if "DateTime" in all_columns else 0
default_target_idx = all_columns.index("Zone 1 Power Consumption") if "Zone 1 Power Consumption" in all_columns else 1

timestamp_col = st.selectbox(
    "Select Timestamp Column",
    all_columns,
    index=default_timestamp_idx
)

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

target_col = st.selectbox(
    "Select Target Column",
    numeric_columns,
    index=numeric_columns.index("Zone 1 Power Consumption") if "Zone 1 Power Consumption" in numeric_columns else 0
)

df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

df = df.dropna(subset=[timestamp_col, target_col]).sort_values(timestamp_col)

st.subheader("Time-Series Cleaning Summary")
st.write({
    "rows_after_cleaning": int(len(df)),
    "min_timestamp": str(df[timestamp_col].min()),
    "max_timestamp": str(df[timestamp_col].max())
})

resample_option = st.selectbox(
    "Optional Resampling",
    ["None", "H", "D"]
)

forecast_horizon = st.number_input(
    "Forecast Horizon (steps ahead)",
    min_value=1,
    max_value=168,
    value=1
)

ts_df = df[[timestamp_col, target_col]].copy()
ts_df = ts_df.set_index(timestamp_col)

if resample_option != "None":
    ts_df = ts_df.resample(resample_option).mean()

ts_df = ts_df.reset_index()

feature_df = ts_df.copy()

feature_df["lag_1"] = feature_df[target_col].shift(1)
feature_df["lag_24"] = feature_df[target_col].shift(24)
feature_df["rolling_mean_24"] = (
    feature_df[target_col]
    .shift(1)
    .rolling(24)
    .mean()
)

feature_df["hour"] = feature_df[timestamp_col].dt.hour
feature_df["weekend"] = feature_df[timestamp_col].dt.dayofweek >= 5
feature_df["month"] = feature_df[timestamp_col].dt.month

feature_df["y_target"] = feature_df[target_col].shift(-forecast_horizon)

feature_df = feature_df.dropna()

X = feature_df[[
    "lag_1",
    "lag_24",
    "rolling_mean_24",
    "hour",
    "weekend",
    "month"
]]

y = feature_df["y_target"]

st.subheader("Feature Table Preview")
st.dataframe(feature_df.head(10), use_container_width=True)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(feature_df[timestamp_col].head(500), feature_df[target_col].head(500))
ax.set_title("Target Preview")
st.pyplot(fig)

# =========================================================
# STUDENT ADDITIONS — MODELING
# Add your forecasting models here.
# Example:
# - train/test split
# - metrics table
# - forecasting pipeline
# =========================================================

st.subheader("STUDENT ADDITIONS — MODELING")
st.code(
    """
# Add your forecasting models here

# Example:
# results_df = pd.DataFrame({
#     "Model": ["Example"],
#     "MAE": [0.0],
#     "RMSE": [0.0]
# })
""",
    language="python"
)

results_df = None

# =========================================================
# STUDENT ADDITIONS — DASHBOARD
# Add extra charts, KPIs, and insights here.
# =========================================================

st.subheader("STUDENT ADDITIONS — DASHBOARD")
st.code(
    """
# Add extra dashboard charts and KPIs here
""",
    language="python"
)

def build_submission_json():
    payload = {
        "student_name": student_name,
        "student_id": student_id,
        "project_title": project_title,
        "project_goal": project_goal,
        "deployed_url": deployed_url,
        "timestamp_column": timestamp_col,
        "target_column": target_col,
        "rows_used": int(len(feature_df)),
        "forecast_horizon": int(forecast_horizon),
        "resampling": resample_option,
        "has_metrics_table": isinstance(results_df, pd.DataFrame),
        "results_table": [] if results_df is None else results_df.to_dict(orient="records"),
        "features": list(X.columns),
        "insights": []
    }
    return payload

submission_payload = build_submission_json()

submission_json_str = json.dumps(submission_payload, indent=2)

project_card = f"""
# Mini Project B

## Student
- Name: {student_name}
- ID: {student_id}

## Project
- Title: {project_title}
- Goal: {project_goal}

## Dataset
- Rows used: {len(feature_df)}
- Timestamp column: {timestamp_col}
- Target column: {target_col}

## Features
{", ".join(X.columns)}
"""

st.subheader("Export Files")

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

st.subheader("AI Grader (/80)")

api_key = None

try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    api_key = st.text_input(
        "Enter OpenRouter API Key",
        type="password"
    )

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
