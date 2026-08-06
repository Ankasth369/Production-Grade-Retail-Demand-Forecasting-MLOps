import json
import os
import subprocess
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ─── Config ──────────────────────────────────────────────────────────────

ARTIFACTS_DIR = Path(
    os.getenv("ARTIFACTS_DIR", Path(__file__).resolve().parent.parent / "artifacts")
)
DATA_DIR = Path(
    os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data")
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DRIFT_LOG_DIR = ARTIFACTS_DIR / "drift_logs"
EVIDENTLY_REPORTS_DIR = ARTIFACTS_DIR / "evidently_reports"
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Demand Forecast MLOps",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    .main .block-container { padding-top: 1rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.25rem;
        border-radius: 0.75rem;
        box-shadow: 0 4px 12px rgba(102,126,234,.25);
    }
    div[data-testid="stMetric"] label { color: rgba(255,255,255,.85) !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #fff !important; font-weight: 700 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] { color: rgba(255,255,255,.8) !important; }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] label { color: #94a3b8 !important; }
    .status-healthy { color: #22c55e; font-weight: 700; }
    .status-drift   { color: #ef4444; font-weight: 700; }
    .status-offline { color: #f59e0b; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Cached data loaders ────────────────────────────────────────────────


@st.cache_data(ttl=300)
def load_training_data():
    path = DATA_DIR / "train.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return None


@st.cache_data(ttl=60)
def load_metrics():
    path = ARTIFACTS_DIR / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_drift_logs():
    if not DRIFT_LOG_DIR.exists():
        return []
    logs = []
    for f in sorted(DRIFT_LOG_DIR.glob("drift_*.json")):
        with open(f) as fh:
            logs.append(json.load(fh))
    return logs


def load_evidently_reports():
    if not EVIDENTLY_REPORTS_DIR.exists():
        return [], []
    html_reports = sorted(EVIDENTLY_REPORTS_DIR.glob("drift_report_*.html"), reverse=True)
    json_summaries = []
    for f in sorted(EVIDENTLY_REPORTS_DIR.glob("drift_summary_*.json"), reverse=True):
        with open(f) as fh:
            json_summaries.append(json.load(fh))
    return html_reports, json_summaries


def load_retrain_meta():
    path = ARTIFACTS_DIR / "retrain_meta.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def api_health():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def api_predict(store_id, item_id, date_str):
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"store_id": store_id, "item_id": item_id, "date": date_str},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"Error {resp.status_code}: {resp.text}"
    except requests.ConnectionError:
        return None, "Cannot connect to API"


# ─── Sidebar ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# \U0001f4ca Demand Forecast")
    st.caption("MLOps Dashboard")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "\U0001f3e0 Overview",
            "\U0001f52e Forecast Explorer",
            "\U0001f4c8 Historical Analysis",
            "\U0001f50d Drift Monitoring",
            "⚙️ Model & System",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    health = api_health()
    if health:
        st.markdown('<p class="status-healthy">● API Online</p>', unsafe_allow_html=True)
        st.caption(f"SMAPE {health['metrics']['smape']:.2f}% • MAE {health['metrics']['mae']:.2f}")
    else:
        st.markdown('<p class="status-offline">● API Offline</p>', unsafe_allow_html=True)

    logs = load_drift_logs()
    if logs and logs[-1]["drift_detected"]:
        st.markdown('<p class="status-drift">● Drift Detected</p>', unsafe_allow_html=True)
    elif logs:
        st.markdown('<p class="status-healthy">● No Drift</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("10 stores × 50 items • XGBoost")

# ═══════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ═══════════════════════════════════════════════════════════════════════

if page == "\U0001f3e0 Overview":
    st.title("Dashboard Overview")

    metrics = load_metrics()
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SMAPE", f"{metrics['smape']:.2f}%")
        c2.metric("MAE", f"{metrics['mae']:.2f}")
        c3.metric("RMSE", f"{metrics['rmse']:.2f}")
        c4.metric("Training Rows", f"{metrics['train_rows']:,}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Model Status")
        if health and metrics:
            st.success("Model loaded and serving predictions")
            st.markdown(f"**Cutoff:** {metrics.get('cutoff_date', 'N/A')}")
            st.markdown(f"**Holdout rows:** {metrics.get('holdout_rows', 'N/A'):,}")
        else:
            st.warning("API not reachable — start the server with `uvicorn app.main:app`")

        retrain = load_retrain_meta()
        if retrain:
            st.markdown(f"**Last retrain:** {retrain['last_retrain'][:19]}")
            st.markdown(f"**Status:** `{retrain['status']}`")

    with col_right:
        st.subheader("Drift Summary")
        if logs:
            drift_count = sum(1 for l in logs if l["drift_detected"])
            d1, d2 = st.columns(2)
            d1.metric("Checks", len(logs))
            d2.metric("Drift Events", drift_count)
            if logs[-1]["drift_detected"]:
                st.error("Latest check detected drift")
            else:
                st.success("Latest check is clean")
        else:
            st.info("No monitoring data yet. Run a drift simulation to get started.")

    st.markdown("---")

    # Quick forecast
    st.subheader("Quick Forecast")
    qc1, qc2, qc3, qc4 = st.columns([1, 1, 1.5, 1])
    qf_store = qc1.number_input("Store", 1, 10, 1, key="qf_s")
    qf_item = qc2.number_input("Item", 1, 50, 1, key="qf_i")
    qf_date = qc3.date_input("Date", key="qf_d")
    with qc4:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Predict", key="qf_btn", type="primary", use_container_width=True):
            res, err = api_predict(qf_store, qf_item, str(qf_date))
            if res:
                st.success(f"**{res['forecast']:.2f} units**")
            else:
                st.error(err)

    # Dataset snapshot
    df = load_training_data()
    if df is not None:
        st.markdown("---")
        st.subheader("Dataset at a Glance")
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Records", f"{len(df):,}")
        g2.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} → {df['date'].max().strftime('%Y-%m-%d')}")
        g3.metric("Stores", int(df["store"].nunique()))
        g4.metric("Items", int(df["item"].nunique()))

        daily = df.groupby("date")["sales"].sum().reset_index()
        fig = px.area(
            daily, x="date", y="sales",
            labels={"sales": "Total Daily Sales", "date": ""},
        )
        fig.update_layout(
            template="plotly_white", height=280, margin=dict(l=0, r=0, t=10, b=0),
            showlegend=False,
        )
        fig.update_traces(line_color="#667eea", fillcolor="rgba(102,126,234,0.15)")
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
#  FORECAST EXPLORER
# ═══════════════════════════════════════════════════════════════════════

elif page == "\U0001f52e Forecast Explorer":
    st.title("Forecast Explorer")

    tab_single, tab_multi, tab_compare, tab_heatmap = st.tabs(
        ["Single Day", "Multi-Day", "Store Comparison", "Demand Heatmap"]
    )

    # --- Single day ---
    with tab_single:
        c1, c2, c3 = st.columns(3)
        sf_store = c1.number_input("Store ID", 1, 10, 1, key="sf_s")
        sf_item = c2.number_input("Item ID", 1, 50, 1, key="sf_i")
        sf_date = c3.date_input("Forecast Date", key="sf_d")

        if st.button("Get Forecast", key="sf_btn", type="primary"):
            res, err = api_predict(sf_store, sf_item, str(sf_date))
            if res:
                r1, r2 = st.columns([1, 2])
                r1.metric("Forecast", f"{res['forecast']:.2f} units")
                r2.json(res)
            else:
                st.error(err)

    # --- Multi-day ---
    with tab_multi:
        st.markdown("Generate a forecast curve for multiple days.")
        c1, c2, c3, c4 = st.columns(4)
        md_store = c1.number_input("Store ID", 1, 10, 1, key="md_s")
        md_item = c2.number_input("Item ID", 1, 50, 1, key="md_i")
        md_start = c3.date_input("Start Date", key="md_sd")
        md_days = c4.number_input("Days Ahead", 1, 90, 14, key="md_n")

        if st.button("Generate Forecast", key="md_btn", type="primary"):
            dates = [md_start + timedelta(days=i) for i in range(md_days)]
            forecasts = []
            bar = st.progress(0, text="Forecasting...")
            for i, d in enumerate(dates):
                res, _ = api_predict(md_store, md_item, str(d))
                if res:
                    forecasts.append({"date": d, "forecast": res["forecast"]})
                bar.progress((i + 1) / len(dates), text=f"Day {i + 1}/{md_days}")
            bar.empty()

            if forecasts:
                fc = pd.DataFrame(forecasts)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fc["date"], y=fc["forecast"], mode="lines+markers",
                    name="Forecast",
                    line=dict(color="#667eea", width=2.5),
                    marker=dict(size=5),
                    fill="tozeroy", fillcolor="rgba(102,126,234,0.1)",
                ))
                fig.update_layout(
                    title=f"Forecast — Store {md_store}, Item {md_item}",
                    template="plotly_white", height=420,
                    xaxis_title="", yaxis_title="Predicted Sales",
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mean", f"{fc['forecast'].mean():.2f}")
                m2.metric("Min", f"{fc['forecast'].min():.2f}")
                m3.metric("Max", f"{fc['forecast'].max():.2f}")
                m4.metric("Total", f"{fc['forecast'].sum():.0f}")

                st.download_button(
                    "⬇ Download CSV", fc.to_csv(index=False),
                    f"forecast_s{md_store}_i{md_item}.csv", "text/csv",
                )

    # --- Store comparison ---
    with tab_compare:
        st.markdown("Compare forecasts across all 10 stores for one item/date.")
        c1, c2 = st.columns(2)
        sc_item = c1.number_input("Item ID", 1, 50, 1, key="sc_i")
        sc_date = c2.date_input("Date", key="sc_d")

        if st.button("Compare Stores", key="sc_btn", type="primary"):
            results = []
            bar = st.progress(0)
            for s in range(1, 11):
                res, _ = api_predict(s, sc_item, str(sc_date))
                if res:
                    results.append({"Store": f"Store {s}", "Forecast": res["forecast"]})
                bar.progress(s / 10)
            bar.empty()

            if results:
                sc_df = pd.DataFrame(results)
                fig = px.bar(
                    sc_df, x="Store", y="Forecast",
                    title=f"Item {sc_item} — {sc_date}",
                    color="Forecast", color_continuous_scale="Viridis",
                    text_auto=".1f",
                )
                fig.update_layout(template="plotly_white", height=420)
                st.plotly_chart(fig, use_container_width=True)

    # --- Demand heatmap ---
    with tab_heatmap:
        st.markdown("Forecast heatmap across stores and items for a single date.")
        hm_date = st.date_input("Date", key="hm_d")
        hm_items = st.slider("Item range", 1, 50, (1, 10), key="hm_ir")

        if st.button("Generate Heatmap", key="hm_btn", type="primary"):
            item_range = list(range(hm_items[0], hm_items[1] + 1))
            total = 10 * len(item_range)
            data = []
            bar = st.progress(0, text="Building heatmap...")
            done = 0
            for s in range(1, 11):
                for it in item_range:
                    res, _ = api_predict(s, it, str(hm_date))
                    data.append({
                        "Store": f"S{s}",
                        "Item": f"I{it}",
                        "Forecast": res["forecast"] if res else np.nan,
                    })
                    done += 1
                    bar.progress(done / total)
            bar.empty()

            hm_df = pd.DataFrame(data)
            pivot = hm_df.pivot(index="Store", columns="Item", values="Forecast")
            fig = px.imshow(
                pivot.values,
                x=pivot.columns.tolist(), y=pivot.index.tolist(),
                color_continuous_scale="YlOrRd", aspect="auto",
                labels=dict(color="Forecast"),
                title=f"Demand Heatmap — {hm_date}",
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
#  HISTORICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

elif page == "\U0001f4c8 Historical Analysis":
    st.title("Historical Analysis")

    df = load_training_data()
    if df is None:
        st.error("Training data not found in `data/train.csv`.")
        st.stop()

    tab_ts, tab_store, tab_season, tab_top = st.tabs(
        ["Sales Trends", "Store Analysis", "Seasonal Patterns", "Top / Bottom Items"]
    )

    # --- Time series ---
    with tab_ts:
        c1, c2 = st.columns(2)
        ts_store = c1.selectbox("Store", sorted(df["store"].unique()), key="ts_s")
        ts_item = c2.selectbox("Item", sorted(df["item"].unique()), key="ts_i")

        series = df[(df["store"] == ts_store) & (df["item"] == ts_item)].sort_values("date").copy()
        series["rolling_7"] = series["sales"].rolling(7).mean()
        series["rolling_28"] = series["sales"].rolling(28).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=series["date"], y=series["sales"], mode="lines",
            name="Daily", line=dict(color="#cbd5e1", width=0.8),
        ))
        fig.add_trace(go.Scatter(
            x=series["date"], y=series["rolling_7"], mode="lines",
            name="7-day MA", line=dict(color="#667eea", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=series["date"], y=series["rolling_28"], mode="lines",
            name="28-day MA", line=dict(color="#ef4444", width=2),
        ))
        fig.update_layout(
            title=f"Store {ts_store}, Item {ts_item}",
            template="plotly_white", height=450,
            xaxis_title="", yaxis_title="Sales",
            hovermode="x unified", legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean", f"{series['sales'].mean():.1f}")
        m2.metric("Median", f"{series['sales'].median():.0f}")
        m3.metric("Std Dev", f"{series['sales'].std():.1f}")
        m4.metric("Max", int(series["sales"].max()))

    # --- Store analysis ---
    with tab_store:
        st.subheader("Average Daily Sales by Store")
        store_avg = df.groupby("store")["sales"].mean().reset_index()
        fig = px.bar(
            store_avg, x="store", y="sales",
            color="sales", color_continuous_scale="Viridis", text_auto=".1f",
            labels={"sales": "Avg Daily Sales", "store": "Store"},
        )
        fig.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Monthly Total Sales by Store")
        df_m = df.copy()
        df_m["yearmonth"] = df_m["date"].dt.to_period("M").astype(str)
        monthly = df_m.groupby(["store", "yearmonth"])["sales"].sum().reset_index()
        fig = px.line(
            monthly, x="yearmonth", y="sales", color="store",
            labels={"sales": "Monthly Sales", "yearmonth": "Month", "store": "Store"},
        )
        fig.update_layout(template="plotly_white", height=420, xaxis=dict(tickangle=-45))
        st.plotly_chart(fig, use_container_width=True)

    # --- Seasonal patterns ---
    with tab_season:
        st.subheader("Day-of-Week Pattern")
        df_dow = df.copy()
        df_dow["dow"] = df_dow["date"].dt.day_name()
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_avg = df_dow.groupby("dow")["sales"].mean().reindex(dow_order).reset_index()
        fig = px.bar(
            dow_avg, x="dow", y="sales",
            color="sales", color_continuous_scale="Blues", text_auto=".1f",
            labels={"sales": "Avg Sales", "dow": ""},
        )
        fig.update_layout(template="plotly_white", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Monthly Seasonality by Year")
        df_my = df.copy()
        df_my["month"] = df_my["date"].dt.month
        df_my["year"] = df_my["date"].dt.year.astype(str)
        month_yr = df_my.groupby(["year", "month"])["sales"].mean().reset_index()
        fig = px.line(
            month_yr, x="month", y="sales", color="year",
            labels={"sales": "Avg Daily Sales", "month": "Month"},
            markers=True,
        )
        fig.update_layout(template="plotly_white", height=400, xaxis=dict(dtick=1))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Year-over-Year Growth")
        yearly = df.groupby(df["date"].dt.year)["sales"].sum().reset_index()
        yearly.columns = ["year", "sales"]
        yearly["yoy"] = yearly["sales"].pct_change() * 100
        fig = px.bar(
            yearly, x="year", y="sales",
            color="sales", color_continuous_scale="Teal",
            text=yearly["yoy"].apply(lambda x: f"+{x:.1f}%" if pd.notna(x) else ""),
            labels={"sales": "Total Sales", "year": "Year"},
        )
        fig.update_layout(template="plotly_white", height=350, showlegend=False)
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    # --- Top / bottom items ---
    with tab_top:
        st.subheader("Top 10 Items by Average Sales")
        item_avg = df.groupby("item")["sales"].mean().reset_index().sort_values("sales", ascending=False)
        top10 = item_avg.head(10)
        bot10 = item_avg.tail(10).sort_values("sales")

        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                top10, x="sales", y="item", orientation="h",
                color="sales", color_continuous_scale="Greens",
                labels={"sales": "Avg Daily Sales", "item": "Item"},
                title="Top 10",
                text_auto=".1f",
            )
            fig.update_layout(template="plotly_white", height=400, showlegend=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.bar(
                bot10, x="sales", y="item", orientation="h",
                color="sales", color_continuous_scale="Reds_r",
                labels={"sales": "Avg Daily Sales", "item": "Item"},
                title="Bottom 10",
                text_auto=".1f",
            )
            fig.update_layout(template="plotly_white", height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sales Distribution Across Items")
        fig = px.box(
            df, x="item", y="sales",
            labels={"sales": "Sales", "item": "Item"},
        )
        fig.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════
#  DRIFT MONITORING
# ═══════════════════════════════════════════════════════════════════════

elif page == "\U0001f50d Drift Monitoring":
    st.title("Drift Monitoring")

    logs = load_drift_logs()

    if not logs:
        st.info("No drift logs yet. Run a simulation to generate data:")
        st.code("python -m scripts.simulate_drift --mode demand_shock", language="bash")
        st.stop()

    drift_count = sum(1 for l in logs if l["drift_detected"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Checks", len(logs))
    c2.metric("Drift Events", drift_count)
    c3.metric("Drift Rate", f"{drift_count / len(logs) * 100:.1f}%")
    latest_status = "⚠️ Drift" if logs[-1]["drift_detected"] else "✅ Stable"
    c4.metric("Latest", latest_status)

    st.markdown("---")

    tab_psi, tab_detail, tab_evidently, tab_timeline, tab_raw = st.tabs(
        ["PSI Timeline", "Feature Deep Dive", "Evidently AI", "Event Timeline", "Raw Reports"]
    )

    all_features = set()
    for l in logs:
        all_features.update(l.get("feature_drift", {}).keys())

    with tab_psi:
        if all_features:
            psi_rows = []
            for l in logs:
                for feat in all_features:
                    fd = l.get("feature_drift", {}).get(feat, {})
                    if "psi" in fd:
                        psi_rows.append({
                            "timestamp": l["timestamp"],
                            "Feature": feat,
                            "PSI": fd["psi"],
                        })
            psi_df = pd.DataFrame(psi_rows)
            psi_df["timestamp"] = pd.to_datetime(psi_df["timestamp"])

            fig = px.line(
                psi_df, x="timestamp", y="PSI", color="Feature",
                title="PSI Score Over Time",
                markers=True,
            )
            fig.add_hline(
                y=0.25, line_dash="dash", line_color="red",
                annotation_text="Drift Threshold (0.25)",
                annotation_position="top left",
            )
            fig.add_hline(
                y=0.1, line_dash="dot", line_color="orange",
                annotation_text="Warning (0.10)",
                annotation_position="bottom left",
            )
            fig.update_layout(template="plotly_white", height=480, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

    with tab_detail:
        sel_feat = st.selectbox("Select Feature", sorted(all_features))
        feat_rows = []
        for l in logs:
            fd = l.get("feature_drift", {}).get(sel_feat, {})
            if fd:
                feat_rows.append({
                    "timestamp": l["timestamp"],
                    "PSI": fd["psi"],
                    "KS Statistic": fd["ks_statistic"],
                    "KS p-value": fd["ks_p_value"],
                    "Drifted": fd["drifted"],
                })
        if feat_rows:
            fd_df = pd.DataFrame(feat_rows)
            latest_f = fd_df.iloc[-1]

            c1, c2, c3 = st.columns(3)
            c1.metric("PSI", f"{latest_f['PSI']:.4f}")
            c2.metric("KS Statistic", f"{latest_f['KS Statistic']:.4f}")
            c3.metric("KS p-value", f"{latest_f['KS p-value']:.6f}")

            if latest_f["Drifted"]:
                st.error(f"⚠️ **{sel_feat}** is currently drifted")
            else:
                st.success(f"✅ **{sel_feat}** is stable")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["PSI", "KS Stat"], y=[latest_f["PSI"], latest_f["KS Statistic"]],
                marker_color=["#667eea", "#764ba2"],
                text=[f"{latest_f['PSI']:.4f}", f"{latest_f['KS Statistic']:.4f}"],
                textposition="outside",
            ))
            fig.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="PSI threshold")
            fig.update_layout(
                template="plotly_white", height=350,
                title=f"{sel_feat} — Latest Scores",
                yaxis_title="Score",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_evidently:
        html_reports, ev_summaries = load_evidently_reports()

        if not html_reports and not ev_summaries:
            st.info(
                "No Evidently reports yet. Run a drift simulation or monitoring to generate them:"
            )
            st.code(
                "python -m scripts.simulate_drift --mode demand_shock", language="bash"
            )
        else:
            if ev_summaries:
                latest = ev_summaries[0]
                c1, c2, c3 = st.columns(3)
                c1.metric(
                    "Dataset Drift",
                    "Yes" if latest.get("dataset_drift") else "No",
                )
                c2.metric(
                    "Drifted Columns",
                    f"{latest.get('n_drifted', 0)}/{latest.get('n_columns', 0)}",
                )
                c3.metric(
                    "Drift Share",
                    f"{latest.get('share_drifted', 0) * 100:.1f}%",
                )

                st.markdown("---")

                if latest.get("column_details"):
                    st.subheader("Per-Column Drift Scores")
                    col_data = []
                    for col, info in latest["column_details"].items():
                        col_data.append({
                            "Column": col,
                            "Drifted": "Yes" if info.get("drift_detected") else "No",
                            "Score": round(info.get("drift_score", 0), 4),
                            "Test": info.get("stattest_name", ""),
                        })
                    st.dataframe(
                        pd.DataFrame(col_data), use_container_width=True, hide_index=True,
                    )

            if html_reports:
                st.markdown("---")
                st.subheader("Interactive Reports")
                selected = st.selectbox(
                    "Select Report",
                    range(len(html_reports)),
                    format_func=lambda i: html_reports[i].stem,
                    key="ev_report_select",
                )
                report_path = html_reports[selected]
                with open(report_path, encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=800, scrolling=True)

    with tab_timeline:
        timeline_data = []
        for l in logs:
            drifted_feats = [f for f, v in l.get("feature_drift", {}).items() if v.get("drifted")]
            timeline_data.append({
                "Timestamp": l["timestamp"][:19],
                "Drift": "✅ No" if not l["drift_detected"] else "⚠️ Yes",
                "Drifted Features": ", ".join(drifted_feats) if drifted_feats else "—",
                "Max PSI": max(
                    (v.get("psi", 0) for v in l.get("feature_drift", {}).values()),
                    default=0,
                ),
            })
        tl_df = pd.DataFrame(timeline_data)
        st.dataframe(tl_df, use_container_width=True, hide_index=True)

    with tab_raw:
        idx = st.selectbox(
            "Select Report",
            range(len(logs)),
            format_func=lambda i: f"{logs[i]['timestamp'][:19]} — {'DRIFT' if logs[i]['drift_detected'] else 'OK'}",
        )
        st.json(logs[idx])

# ═══════════════════════════════════════════════════════════════════════
#  MODEL & SYSTEM
# ═══════════════════════════════════════════════════════════════════════

elif page == "⚙️ Model & System":
    st.title("Model & System")

    tab_info, tab_retrain, tab_sim = st.tabs(
        ["Model Info", "Retrain Pipeline", "Drift Simulation"]
    )

    with tab_info:
        metrics = load_metrics()
        if metrics:
            st.subheader("Current Model Metrics")
            c1, c2, c3 = st.columns(3)
            c1.metric("SMAPE", f"{metrics['smape']:.4f}%")
            c2.metric("MAE", f"{metrics['mae']:.4f}")
            c3.metric("RMSE", f"{metrics['rmse']:.4f}")

            st.markdown("---")

            col_l, col_r = st.columns(2)
            with col_l:
                st.subheader("Training Details")
                st.json(metrics)

            with col_r:
                st.subheader("Feature Inventory")
                _FEATURES = [
                    "store_code", "item_code", "dow", "is_weekend", "month",
                    "weekofyear", "year", "dayofyear",
                    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28", "sales_lag_364",
                    "sales_rmean_7", "sales_rstd_7", "sales_rmean_28", "sales_rstd_28",
                    "sales_rmean_90", "sales_rstd_90",
                ]
                _DRIFT_FEATURES = [
                    "sales_lag_364", "sales_rmean_7", "sales_rmean_28",
                    "month", "sales_rmean_90",
                ]
                feat_info = []
                for f in _FEATURES:
                    feat_info.append({
                        "Feature": f,
                        "Monitored": "\U0001f50d" if f in _DRIFT_FEATURES else "",
                        "Type": (
                            "Lag" if "lag" in f
                            else "Rolling" if "rmean" in f or "rstd" in f
                            else "Calendar" if f in ("dow", "is_weekend", "month", "weekofyear", "year", "dayofyear")
                            else "Identity"
                        ),
                    })
                st.dataframe(pd.DataFrame(feat_info), use_container_width=True, hide_index=True)
        else:
            st.warning("No model trained yet. Run `python -m src.models.train`.")

        st.markdown("---")
        st.subheader("API Controls")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Reload Model", type="primary", use_container_width=True):
                try:
                    resp = requests.post(f"{API_URL}/reload", timeout=10)
                    if resp.status_code == 200:
                        st.success("Model reloaded")
                        st.json(resp.json())
                    else:
                        st.error(f"Reload failed: {resp.status_code}")
                except Exception:
                    st.error("Cannot connect to API")
        with c2:
            if st.button("Health Check", use_container_width=True):
                h = api_health()
                if h:
                    st.success("API is healthy")
                    st.json(h)
                else:
                    st.error("API not reachable")

        retrain = load_retrain_meta()
        if retrain:
            st.markdown("---")
            st.subheader("Last Retrain Result")
            st.json(retrain)

    with tab_retrain:
        st.subheader("Trigger Retraining")
        st.markdown("""
**Pipeline steps:**
1. Train a candidate model on the full dataset
2. Evaluate on the 28-day holdout
3. Promote only if candidate SMAPE ≤ current × 1.05
4. 24-hour cooldown between runs (unless forced)
""")

        force = st.checkbox("Force retrain (skip cooldown check)")

        if st.button("Start Retraining", type="primary"):
            cmd = ["python", "-m", "src.pipelines.retrain"]
            if force:
                cmd.append("--force")
            with st.spinner("Training in progress — this may take a few minutes..."):
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=600,
                        cwd=str(PROJECT_ROOT),
                    )
                    if result.stdout.strip():
                        try:
                            parsed = json.loads(result.stdout.strip())
                            status = parsed.get("status", "unknown")
                            if status == "promoted":
                                st.success(f"Model promoted! New SMAPE: {parsed.get('new_smape')}")
                            elif status == "rejected":
                                st.warning(
                                    f"Candidate rejected. New SMAPE {parsed.get('new_smape')} > "
                                    f"threshold {parsed.get('threshold')}"
                                )
                            elif status == "skipped":
                                st.info(f"Skipped: {parsed.get('reason')}")
                            else:
                                st.info(f"Status: {status}")
                            st.json(parsed)
                        except json.JSONDecodeError:
                            st.code(result.stdout)
                    if result.stderr.strip():
                        st.error(result.stderr)
                except subprocess.TimeoutExpired:
                    st.error("Training timed out after 10 minutes")
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab_sim:
        st.subheader("Drift Simulation")
        st.markdown("Inject synthetic drift to test the monitoring system.")

        mode = st.selectbox("Mode", ["demand_shock", "distribution_shift", "gradual"])
        descriptions = {
            "demand_shock": "Multiply sales by **1.8×** for items 1–5 from Oct 2017 onward",
            "distribution_shift": "Swap sales between high/low demand item pairs",
            "gradual": "Apply **0.2% daily** compounding growth for items 1–10 from Jul 2017",
        }
        st.info(descriptions[mode])

        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running drift simulation..."):
                try:
                    result = subprocess.run(
                        ["python", "-m", "scripts.simulate_drift", "--mode", mode],
                        capture_output=True, text=True, timeout=300,
                        cwd=str(PROJECT_ROOT),
                    )
                    if result.stdout.strip():
                        st.code(result.stdout)
                    if result.returncode == 0:
                        st.success("Simulation complete! Switch to **Drift Monitoring** to see results.")
                    else:
                        st.error(f"Simulation failed:\n{result.stderr}")
                except Exception as e:
                    st.error(f"Error: {e}")
