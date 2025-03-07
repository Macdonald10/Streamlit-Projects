import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# Configuration
DATA_FILE = "covid.json"
COLORS = {
    "cases": "#1f77b4",  # Blue
    "deaths": "#ff7f0e",  # Orange
    "recovered": "#2ca02c",  # Green
    "active": "#d62728",  # Red
    "background": "#f0f2f6",  # Light gray
    "text": "#333333",  # Dark gray
    "white": "#ffffff",  # White
}

# --- Helper Functions ---
@st.cache_data
def load_data():
    """
    Loads data from a JSON file, cleans column names, and converts the 'date' column to datetime.
    """
    try:
        df = pd.read_json(DATA_FILE)
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace("[^a-z0-9]+", "_", regex=True)
        )

        column_mapping = {
            "totalcases": "total_confirmed_cases",
            "confirmed": "total_confirmed_cases",
            "deaths": "total_deaths",
            "recovered": "total_recovered",
            "active": "active_cases",
            "new_cases": "daily_confirmed_cases",
            "new_deaths": "daily_deaths"
        }
        df.rename(columns=column_mapping, inplace=True)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
        else:
            st.error("Date column not found in dataset!")
            return None

        numeric_cols = ["total_confirmed_cases", "total_deaths",
                       "total_recovered", "active_cases",
                       "daily_confirmed_cases", "daily_deaths"]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace("[^0-9]", "", regex=True),
                    errors="coerce"
                ).fillna(0).astype("Int64")

        return df

    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        return None

def calculate_growth_metrics(df):
    """
    Calculates growth metrics such as daily change, growth rate, and doubling time.
    """
    df['daily_change'] = df['total_confirmed_cases'].diff().fillna(0)
    df['growth_rate'] = df['total_confirmed_cases'].pct_change().fillna(0)
    df['growth_rate_pct'] = df['growth_rate'] * 100
    df['doubling_time_days'] = np.log(2) / np.log(1 + df['growth_rate'])
    df['doubling_time_days'] = df['doubling_time_days'].replace([np.inf, -np.inf], 0).fillna(0)  # Handle infinite values
    return df

def calculate_rates(df):
    """
    Calculates case fatality rate and recovery rate.
    """
    df['cfr'] = df['total_deaths'] / df['total_confirmed_cases']
    df['recovery_rate'] = df['total_recovered'] / df['total_confirmed_cases']
    df['cfr_pct'] = df['cfr'] * 100
    df['recovery_rate_pct'] = df['recovery_rate'] * 100
    return df

# --- Main App ---
def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(
        layout="wide",
        page_title="🦠 COVID-19 Dashboard",
        page_icon="🦠"
    )

    # Custom CSS for modern styling
    st.markdown(f"""
        <style>
            /* General styling */
            .stApp {{
                background-color: {COLORS["background"]};
                font-family: 'Arial', sans-serif;
            }}
            .stMetric {{
                background-color: {COLORS["white"]};
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                text-align: center;
            }}
            .stHeader {{
                color: {COLORS["cases"]};
                font-size: 24px;
                font-weight: bold;
                margin-bottom: 10px;
            }}
            .stTabs [data-baseweb="tab-list"] {{
                gap: 10px;
            }}
            .stTabs [data-baseweb="tab"] {{
                padding: 10px 20px;
                border-radius: 5px;
                background-color: {COLORS["white"]};
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
            .stTabs [aria-selected="true"] {{
                background-color: {COLORS["cases"]} !important;
                color: {COLORS["white"]} !important;
            }}
            /* Custom styling for Data analysis and B */
            .section-header {{
                font-size: 32px;
                font-weight: bold;
                color: {COLORS["cases"]};
                margin-bottom: 20px;
            }}
            .section-subheader {{
                font-size: 24px;
                font-weight: bold;
                color: {COLORS["recovered"]};
                margin-bottom: 15px;
            }}
            .section-metric {{
                font-size: 18px;
                font-weight: bold;
                color: {COLORS["text"]};
            }}
            .section-caption {{
                font-size: 14px;
                color: {COLORS["text"]};
            }}
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to:", ["Data analysis", "Advanced Data analysis"])

    df = load_data()  # Call the load_data function
    if df is None:
        st.stop()

    # --- Data analysis: Interactive Explorer ---
    if section == "Data analysis":
        st.markdown('<div class="section-header">🦠 COVID-19 Interactive Explorer</div>', unsafe_allow_html=True)

        # Sidebar filters
        st.sidebar.header("📅 Data Filters")
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        start_date, end_date = date_range if isinstance(date_range, tuple) else (date_range, date_range)
        filtered_df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]

        # --- Comparative Analysis ---
        st.markdown('<div class="section-subheader">Comparative Analysis</div>', unsafe_allow_html=True)
        compare_date = st.date_input("Select Date for Comparison", value=min_date, min_value=min_date, max_value=max_date)

        compare_df = df[(df["date"] == pd.to_datetime(compare_date))]
        if not compare_df.empty:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Cases", f"{filtered_df['total_confirmed_cases'].max():,}")
                change = ((filtered_df['total_confirmed_cases'].max() - compare_df['total_confirmed_cases'].iloc[0]) / compare_df['total_confirmed_cases'].iloc[0]) * 100
                st.markdown(f'<div class="section-caption">{change:.2f}% Change vs {compare_date}</div>', unsafe_allow_html=True)
            with col2:
                st.metric("Fatalities", f"{filtered_df['total_deaths'].max():,}")
                change = ((filtered_df['total_deaths'].max() - compare_df['total_deaths'].iloc[0]) / compare_df['total_deaths'].iloc[0]) * 100
                st.markdown(f'<div class="section-caption">{change:.2f}% Change vs {compare_date}</div>', unsafe_allow_html=True)
            with col3:
                st.metric("Active Cases", f"{filtered_df['active_cases'].max():,}")
                change = ((filtered_df['active_cases'].max() - compare_df['active_cases'].iloc[0]) / compare_df['active_cases'].iloc[0]) * 100
                st.markdown(f'<div class="section-caption">{change:.2f}% Change vs {compare_date}</div>', unsafe_allow_html=True)
        else:
            st.warning("No data available for the comparison date.")

        # Metrics Section
        st.markdown('<div class="section-subheader">📊 Real-time Metrics</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cases", f"{filtered_df['total_confirmed_cases'].max():,}")
        with col2:
            st.metric("Fatalities", f"{filtered_df['total_deaths'].max():,}")
        with col3:
            st.metric("Active Cases", f"{filtered_df['active_cases'].max():,}")
        with col4:
            st.metric("Recoveries", f"{filtered_df['total_recovered'].max():,}")

        # Visualizations in Tabs
        st.markdown('<div class="section-subheader">📈 Visual Analysis</div>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Trends", "Distribution"])

        with tab1:
            colA1, colA2 = st.columns(2)
            with colA1:
                st.subheader("Case Progression Timeline")
                fig_trend = px.line(filtered_df, x="date", y="total_confirmed_cases", title="Total Cases Over Time",
                                  template="plotly_white", color_discrete_sequence=[COLORS["cases"]])
                st.plotly_chart(fig_trend, use_container_width=True)
            with colA2:
                st.subheader("Daily Cases Area Chart")
                fig_area = px.area(filtered_df, x="date", y="daily_confirmed_cases", title="Daily New Cases",
                                 template="plotly_white", color_discrete_sequence=[COLORS["active"]])
                st.plotly_chart(fig_area, use_container_width=True)

        with tab2:
            colB1, colB2 = st.columns(2)
            with colB1:
                st.subheader("Case Distribution Pie Chart")
                latest = filtered_df.iloc[-1]
                pie_data = {"Active": latest["active_cases"], "Recovered": latest["total_recovered"], "Deaths": latest["total_deaths"]}
                fig_pie = px.pie(names=list(pie_data.keys()), values=list(pie_data.values()), hole=0.4,
                               title="Distribution of Cases", template="plotly_white",
                               color_discrete_sequence=[COLORS["active"], COLORS["recovered"], COLORS["deaths"]])
                st.plotly_chart(fig_pie, use_container_width=True)
            with colB2:
                st.subheader("Daily Cases by Day of Week")
                daily_df = filtered_df.groupby(filtered_df["date"].dt.day_name())["daily_confirmed_cases"].mean().reset_index()
                fig_bar = px.bar(daily_df, x="date", y="daily_confirmed_cases", title="Average Daily Cases by Day",
                               template="plotly_white", color_discrete_sequence=[COLORS["cases"]])
                st.plotly_chart(fig_bar, use_container_width=True)

        # Outcome Analysis
        st.markdown('<div class="section-subheader">📊 Outcome Analysis</div>', unsafe_allow_html=True)
        colC1, colC2 = st.columns(2)
        with colC1:
            st.subheader("Case Fatality Rate")
            mortality_rate = (filtered_df["total_deaths"].max() / filtered_df["total_confirmed_cases"].max()) * 100
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=mortality_rate,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Mortality Rate (%)"},
                gauge={"axis": {"range": [0, 100]}}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
        with colC2:
            st.subheader("Recovery Rate")
            recovery_rate = (filtered_df["total_recovered"].max() / filtered_df["total_confirmed_cases"].max()) * 100
            fig_gauge_recovery = go.Figure(go.Indicator(
                mode="gauge+number",
                value=recovery_rate,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Recovery Rate (%)"},
                gauge={"axis": {"range": [0, 100]}}
            ))
            st.plotly_chart(fig_gauge_recovery, use_container_width=True)

    # --- Advanced Data analysis: Advanced Analytics ---
    elif section == "Advanced Data analysis":
        st.markdown('<div class="section-header">🧪 Advanced COVID-19 Analytics Dashboard</div>', unsafe_allow_html=True)

        # --- Data Processing ---
        df = calculate_growth_metrics(df)
        df = calculate_rates(df)

        # Scenario Planning
        st.markdown('<div class="section-subheader">Scenario Planning</div>', unsafe_allow_html=True)
        projection_days = st.slider("Projection Horizon (Days)", 7, 60, 30)

        # Data manipulation: fill missing values with 0
        df['daily_confirmed_cases'] = df['daily_confirmed_cases'].fillna(0)

        # Calculate simple projection
        last_week_avg = df['daily_confirmed_cases'].tail(7).mean()
        projected_cases = [last_week_avg * (1.02)**i for i in range(projection_days)]  # Assuming a 2% daily growth

        # Projection Graph
        st.subheader("Projected Cases")
        projection_dates = [df['date'].iloc[-1] + pd.Timedelta(days=i) for i in range(projection_days)]
        fig_projection = px.line(x=projection_dates, y=projected_cases, title="Projected Cases (Simple Model)",
                               template="plotly_white", color_discrete_sequence=[COLORS["cases"]])
        st.plotly_chart(fig_projection, use_container_width=True)

        # --- Metrics ---
        st.markdown('<div class="section-subheader">Key Metrics</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Peak Daily Cases", f"{df['daily_confirmed_cases'].max():,}")
        with col2:
            st.metric("Current CFR", f"{df['cfr_pct'].iloc[-1]:.2f}%")
        with col3:
            st.metric("Recovery Rate", f"{df['recovery_rate_pct'].iloc[-1]:.2f}%")

        # --- Visualizations in Tabs ---
        st.markdown('<div class="section-subheader">Advanced Visualizations</div>', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(["Trend Analysis", "Phase Breakdown", "Advanced Metrics"])

        with tab1:
            colB1, colB2 = st.columns(2)
            with colB1:
                st.subheader("Cumulative Trends (Log Scale)")
                fig_cumulative = px.line(df, x="date", y=["total_confirmed_cases", "total_deaths", "total_recovered"],
                                      title="Cumulative Trends (Log Scale)", log_y=True, template="plotly_white",
                                      color_discrete_sequence=[COLORS["cases"], COLORS["deaths"], COLORS["recovered"]])
                st.plotly_chart(fig_cumulative, use_container_width=True)
            with colB2:
                st.subheader("Daily Cases with 7-Day Moving Averages")
                fig_daily = px.area(df, x="date", y=["daily_confirmed_cases", "daily_deaths"],
                                     title="Daily Cases with 7-Day Moving Averages", template="plotly_white",
                                     color_discrete_sequence=[COLORS["cases"], COLORS["deaths"]])
                st.plotly_chart(fig_daily, use_container_width=True)

        with tab2:
            colC1, colC2 = st.columns(2)
            with colC1:
                st.subheader("Growth Phase Analysis")
                fig_phase = px.scatter(df, x="date", y="growth_rate_pct",
                                        size="daily_confirmed_cases",
                                        title="Growth Phase Analysis", template="plotly_white",
                                        color_discrete_sequence=[COLORS["cases"]])
                st.plotly_chart(fig_phase, use_container_width=True)
            with colC2:
                st.subheader("Phase Summary Statistics")
                phase_summary = df.groupby(df["growth_rate_pct"]).agg({
                    'daily_confirmed_cases': ['mean', 'max'],
                    'doubling_time_days': ['mean','max']
                }).style.format("{:.2f}")
                st.dataframe(phase_summary, use_container_width=True)

        with tab3:
            colD1, colD2 = st.columns(2)
            with colD1:
                st.subheader("Distribution Analysis")
                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(x=df["daily_confirmed_cases"], name="Daily Cases"))
                fig_dist.add_trace(go.Histogram(x=df["daily_deaths"], name="Daily Deaths"))
                fig_dist.update_layout(barmode="overlay", title="Distribution of Daily Cases & Deaths",
                                     template="plotly_white")
                st.plotly_chart(fig_dist, use_container_width=True)
            with colD2:
                st.subheader("Advanced Metrics Correlation")
                correlation_matrix = df[["growth_rate_pct", "doubling_time_days", "cfr_pct", "recovery_rate_pct"]].corr()
                fig_heatmap = go.Figure(data=go.Heatmap(z=correlation_matrix.values,
                                                        x=correlation_matrix.columns,
                                                        y=correlation_matrix.columns,
                                                        colorscale='Viridis'))
                fig_heatmap.update_layout(title="Correlation Matrix of Advanced Metrics", template="plotly_white")
                st.plotly_chart(fig_heatmap, use_container_width=True)

if __name__ == "__main__":
    main()
