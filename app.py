"""
Enterprise Customer Feedback Intelligence Dashboard.

Features:
- Asynchronous batch ingestion with live polling progress
- Deep-dive Aspect-Based Sentiment Analysis (ABSA)
- Real-time urgency & food safety triage inbox
- Executive AI Operational Briefing memo synthesis
- Unsupervised K-Means & PCA Semantic Topic Clustering (Plotly)
- Semantic Cache performance & cost-saving telemetry
- Historical analytics & CSV/JSON export
"""
import os
import threading
import time
import json
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import uvicorn

from services.executive_brief import brief_generator
from services.clustering import clusterer

# Configure page layout
st.set_page_config(
    page_title="Feedback Intelligence Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


# Auto-start FastAPI backend if running locally or on single-service platforms (e.g. Streamlit Cloud)
@st.cache_resource
def start_embedded_backend():
    def _is_online():
        try:
            return requests.get(f"{API_BASE_URL}/api/v1/health", timeout=1.0).status_code == 200
        except Exception:
            return False

    if _is_online():
        return True

    if "127.0.0.1" in API_BASE_URL or "localhost" in API_BASE_URL:
        def _run_server():
            import api
            uvicorn.run(api.app, host="127.0.0.1", port=8000, log_level="warning")

        thread = threading.Thread(target=_run_server, daemon=True)
        thread.start()
        for _ in range(20):
            time.sleep(0.5)
            if _is_online():
                break
    return True


start_embedded_backend()


# Helper functions to talk to FastAPI backend
def check_api_health():
    try:
        res = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=1.5)
        if res.status_code == 200:
            return True, res.json()
    except Exception:
        pass
    return False, None


def get_metrics():
    try:
        res = requests.get(f"{API_BASE_URL}/api/v1/metrics", timeout=2.0)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {}


# -------------------------------------------------------------
# SIDEBAR: Architecture, Telemetry, and Actions
# -------------------------------------------------------------
with st.sidebar:
    st.title("⚡ System Radar")
    
    is_healthy, health_data = check_api_health()
    if is_healthy:
        st.success("🟢 API Gateway: Online")
        mode = health_data.get("llm_mode", "Unknown")
        st.caption(f"**Inference Engine**: `{mode}`")
    else:
        st.error("🔴 API Gateway: Offline")
        st.caption("Start the backend in terminal 1:")
        st.code("uv run uvicorn api:app --reload", language="bash")

    st.markdown("---")
    st.subheader("💡 Semantic Cache Telemetry")
    metrics_data = get_metrics()
    sys_metrics = metrics_data.get("system_metrics", {})
    
    col_sb1, col_sb2 = st.columns(2)
    col_sb1.metric("Cache Hit Rate", f"{sys_metrics.get('hit_rate_pct', 0.0)}%")
    col_sb2.metric("Saved ($)", f"${sys_metrics.get('estimated_cost_saved_usd', 0.0):.4f}")
    st.caption(f"Cached entries: {sys_metrics.get('cached_entries', 0)} | Total hits: {sys_metrics.get('hits', 0)}")

    if st.button("🧹 Clear Semantic Cache"):
        try:
            requests.post(f"{API_BASE_URL}/api/v1/cache/clear", timeout=2.0)
            st.success("Cache invalidated!")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear cache: {e}")

    st.markdown("---")
    st.subheader("📂 Quick Load Demo")
    if st.button("Load Sample Dataset (CSV)"):
        try:
            sample_df = pd.read_csv("sample_batch.csv")
            st.session_state.batch_input_text = "\n".join(sample_df["review"].dropna().tolist())
            st.toast("Loaded 20 realistic multi-aspect reviews!", icon="✅")
        except Exception as e:
            st.error(f"Could not load sample CSV: {e}")


# -------------------------------------------------------------
# MAIN APP HEADER
# -------------------------------------------------------------
st.title("🛡️ Enterprise Customer Feedback Intelligence Hub")
st.markdown(
    "High-throughput, asynchronous sentiment analytics with **Aspect-Based Sentiment Analysis (ABSA)**, "
    "**Semantic Vector Caching**, **Topic Clustering**, and **Urgency Triage**."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Bulk Batch Pipeline",
    "🔍 Single Review Deep Dive",
    "🗺️ Semantic Topic Clusters",
    "📊 Audit History & Analytics"
])


# =============================================================
# TAB 1: BATCH INGESTION PIPELINE
# =============================================================
with tab1:
    st.subheader("High-Throughput Batch Ingestion")
    st.write("Upload a CSV file or paste bulk reviews to dispatch non-blocking background workers.")

    uploaded_file = st.file_uploader("Upload CSV or TXT (reviews column will be extracted)", type=["csv", "txt"])
    
    if "batch_input_text" not in st.session_state:
        st.session_state.batch_input_text = ""

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
                col = next((c for c in df_upload.columns if "review" in c.lower() or "text" in c.lower()), df_upload.columns[0])
                st.session_state.batch_input_text = "\n".join(df_upload[col].astype(str).tolist())
            else:
                st.session_state.batch_input_text = uploaded_file.read().decode("utf-8")
            st.success(f"Loaded file with {len(st.session_state.batch_input_text.splitlines())} items.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    reviews_input = st.text_area(
        "Raw Reviews (one per line):",
        value=st.session_state.batch_input_text,
        height=160,
        placeholder="Paste 10 to 5,000 reviews here...",
    )

    if st.button("⚡ Dispatch Distributed Job", type="primary"):
        reviews = [line.strip() for line in reviews_input.split("\n") if line.strip()]
        if not reviews:
            st.warning("Please provide at least one review.")
        elif not is_healthy:
            st.error("Cannot dispatch job: Backend API is offline. Start the backend first.")
        else:
            try:
                with st.spinner("Submitting batch to queue..."):
                    resp = requests.post(
                        f"{API_BASE_URL}/api/v1/jobs",
                        json={"reviews": reviews},
                        timeout=5.0,
                    )
                if resp.status_code == 202:
                    job_data = resp.json()
                    job_id = job_data["job_id"]
                    st.info(f"✅ Job Enqueued: `{job_id}` | Total: {job_data['total']} reviews")

                    # Live Polling Progress Bar
                    progress_bar = st.progress(0, text="Worker nodes executing...")
                    status_text = st.empty()

                    completed = 0
                    total = job_data["total"]
                    results_data = []

                    start_time = time.time()
                    while True:
                        time.sleep(0.3)
                        poll_resp = requests.get(f"{API_BASE_URL}/api/v1/jobs/{job_id}", timeout=5.0)
                        if poll_resp.status_code == 200:
                            data = poll_resp.json()
                            completed = data["completed"] + data["failed"]
                            pct = min(completed / total if total > 0 else 1.0, 1.0)
                            elapsed = time.time() - start_time
                            rate = completed / elapsed if elapsed > 0 else 0
                            
                            progress_bar.progress(pct, text=f"Processing {completed}/{total} reviews ({rate:.1f} reviews/sec)")
                            
                            if data["status"] in ["completed", "failed"] or completed >= total:
                                results_data = data.get("results", [])
                                break

                    progress_bar.progress(1.0, text="✨ Batch Processing Complete!")
                    st.session_state.latest_batch_results = results_data
                    st.session_state.current_job_id = job_id
                else:
                    st.error(f"Job submission failed: {resp.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

    # Display Batch Results
    if "latest_batch_results" in st.session_state and st.session_state.latest_batch_results:
        batch_results = st.session_state.latest_batch_results
        st.markdown("---")
        st.subheader("📊 Batch Intelligence Insights")

        # Top KPI Metrics
        scores = [r["overall_score"] for r in batch_results]
        pos_count = sum(1 for r in batch_results if r["overall_label"] == "positive")
        critical_alerts = [r for r in batch_results if r.get("urgency") in ["high", "critical"]]
        cached_count = sum(1 for r in batch_results if r.get("cached", False))

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Reviews", len(batch_results))
        kpi2.metric("Average Rating", f"{sum(scores)/len(scores):.1f} / 5" if scores else "N/A")
        kpi3.metric("Positive %", f"{(pos_count/len(batch_results)*100):.0f}%")
        kpi4.metric("Cache Hits", f"{cached_count} ({cached_count/len(batch_results)*100:.0f}%)")
        kpi5.metric("Urgent Alerts", len(critical_alerts), delta=f"{len(critical_alerts)} need review", delta_color="inverse")

        # Critical Alerts Feed
        if critical_alerts:
            st.error("🚨 Critical / Urgent Operational Alerts Detected:")
            for alert in critical_alerts:
                with st.expander(f"⚠️ [{alert.get('urgency').upper()}] {alert.get('theme', '').title()} - \"{alert['text'][:80]}...\""):
                    st.write(f"**Full Text**: {alert['text']}")
                    st.write(f"**Root Cause**: {alert.get('root_cause') or 'N/A'}")
                    st.write(f"**Recommended Action**: {alert.get('suggested_action') or 'N/A'}")

        # Executive Operational Briefing Action
        col_memo_btn, col_memo_space = st.columns([2, 3])
        with col_memo_btn:
            if st.button("📋 Synthesize Executive Operational Brief", type="secondary"):
                with st.spinner("Synthesizing executive briefing memo..."):
                    brief_resp = brief_generator.generate_brief(batch_results)
                    st.session_state.executive_brief_memo = brief_resp.get("markdown", "")

        if "executive_brief_memo" in st.session_state and st.session_state.executive_brief_memo:
            with st.container(border=True):
                st.markdown(st.session_state.executive_brief_memo)

        # Summary Table
        st.subheader("Structured Review Records")
        table_rows = []
        for r in batch_results:
            aspect_summary = ", ".join([f"{a['aspect']}: {a['sentiment']}({a['score']})" for a in r.get("aspects", [])])
            table_rows.append({
                "Review": r["text"],
                "Sentiment": r["overall_label"],
                "Rating": r["overall_score"],
                "Theme": r["theme"],
                "Urgency": r.get("urgency", "low"),
                "Aspects": aspect_summary or "general",
                "Cached": "⚡ Yes" if r.get("cached") else "No",
                "Source": r.get("source", "llm"),
                "Latency (ms)": r.get("latency_ms", 0.0),
            })
        df_results = pd.DataFrame(table_rows)
        st.dataframe(df_results, use_container_width=True)

        # Export CSV Button
        csv_bytes = df_results.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Analysis Report (CSV)", data=csv_bytes, file_name="feedback_analysis_report.csv", mime="text/csv")


# =============================================================
# TAB 2: SINGLE REVIEW DEEP DIVE (ABSA)
# =============================================================
with tab2:
    st.subheader("Deep-Dive Aspect-Based Analysis")
    st.write("Inspect multi-dimensional customer sentiments, evidence snippets, and root-cause diagnoses.")

    single_input = st.text_input(
        "Enter a customer review:",
        value="The food was delicious and fresh, but the delivery took over an hour and the courier was rude.",
    )

    col_btn, col_check = st.columns([1, 3])
    with col_btn:
        analyze_single_btn = st.button("Analyze Review", type="primary")
    with col_check:
        bypass_cache = st.checkbox("Bypass Semantic Cache (Force Live LLM)")

    if analyze_single_btn:
        if not single_input.strip():
            st.warning("Please type a review.")
        elif not is_healthy:
            st.error("Backend API is offline.")
        else:
            try:
                with st.spinner("Analyzing with multi-aspect intelligence..."):
                    res = requests.post(
                        f"{API_BASE_URL}/api/v1/analyze?bypass_cache={str(bypass_cache).lower()}",
                        json={"text": single_input},
                        timeout=10.0,
                    )
                if res.status_code == 200:
                    data = res.json()
                    st.success("Analysis Complete!")

                    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                    col_res1.metric("Overall Label", data["overall_label"].upper())
                    col_res2.metric("Overall Score", f"{data['overall_score']} / 5")
                    col_res3.metric("Urgency", data.get("urgency", "low").upper())
                    col_res4.metric("Latency", f"{data.get('latency_ms', 0)} ms", delta="Cached" if data.get("cached") else "Fresh")

                    # Aspect Table
                    st.subheader("Aspect-Based Sentiments (ABSA)")
                    aspects = data.get("aspects", [])
                    if aspects:
                        df_aspects = pd.DataFrame(aspects)
                        st.table(df_aspects)
                    else:
                        st.info("No distinct operational aspects detected.")

                    # Operational Diagnosis
                    st.subheader("Operational Diagnosis & Action Plan")
                    col_diag1, col_diag2 = st.columns(2)
                    with col_diag1:
                        st.markdown("**Root Cause:**")
                        st.info(data.get("root_cause") or "No operational failure identified.")
                    with col_diag2:
                        st.markdown("**Recommended Management Action:**")
                        st.warning(data.get("suggested_action") or "Standard customer engagement.")
                else:
                    st.error(f"Analysis failed: {res.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")


# =============================================================
# TAB 3: SEMANTIC TOPIC CLUSTERING (PLOTLY)
# =============================================================
with tab3:
    st.subheader("🗺️ Unsupervised Customer Problem Space (Topic Clustering)")
    st.markdown("Groups customer feedback into distinct operational clusters using vector embeddings, **K-Means**, and **PCA 2D projection**.")

    data_source_opt = st.radio("Cluster Dataset Source:", ["Current Batch Results", "Database History (up to 100 reviews)"], horizontal=True)
    
    target_data = []
    if data_source_opt == "Current Batch Results" and "latest_batch_results" in st.session_state:
        target_data = st.session_state.latest_batch_results
    elif is_healthy:
        try:
            hist_res = requests.get(f"{API_BASE_URL}/api/v1/history?limit=100", timeout=3.0)
            if hist_res.status_code == 200:
                target_data = hist_res.json()
        except Exception:
            pass

    if not target_data:
        st.info("No data available to cluster. Please analyze a batch or load sample reviews in Tab 1.")
    else:
        clustering_result = clusterer.cluster_reviews(target_data, max_clusters=4)
        points = clustering_result.get("points", [])
        cluster_summaries = clustering_result.get("clusters", [])

        if points:
            # Cluster metrics row
            st.markdown("#### Operational Cluster Diagnostics")
            cols_cl = st.columns(len(cluster_summaries))
            for i, c in enumerate(cluster_summaries):
                with cols_cl[i]:
                    st.metric(
                        label=c["name"],
                        value=f"{c['size']} reviews",
                        delta=f"Avg: {c['average_score']}/5 | {c['negative_reviews']} neg",
                        delta_color="off" if c['negative_reviews'] == 0 else "inverse"
                    )

            # Interactive Plotly Scatter Plot
            df_points = pd.DataFrame(points)
            fig = px.scatter(
                df_points,
                x="x",
                y="y",
                color="cluster_name",
                symbol="sentiment",
                hover_name="cluster_name",
                hover_data={"review": True, "score": True, "urgency": True, "x": False, "y": False},
                title="2D Vector Space Projection of Customer Sentiment & Topics",
                template="plotly_white",
                height=520,
            )
            fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(legend_title_text="Topic Cluster")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough reviews to perform topic clustering.")


# =============================================================
# TAB 4: AUDIT HISTORY & AGGREGATE ANALYTICS
# =============================================================
with tab4:
    st.subheader("Historical Analytics & Audit Trail")
    
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        sentiment_filter = st.selectbox("Filter Sentiment:", ["All", "positive", "negative", "neutral"])
    with col_f2:
        search_query = st.text_input("Search reviews by keyword:", placeholder="e.g. biryani, late, crash")

    if is_healthy:
        try:
            url = f"{API_BASE_URL}/api/v1/history?limit=100"
            if sentiment_filter != "All":
                url += f"&label={sentiment_filter}"
            
            hist_res = requests.get(url, timeout=3.0)
            if hist_res.status_code == 200:
                history = hist_res.json()
                if search_query:
                    history = [h for h in history if search_query.lower() in h["review"].lower()]

                if history:
                    st.write(f"Total matching records: **{len(history)}**")
                    df_hist = pd.DataFrame([
                        {
                            "Date": h.get("created_at", "")[:19].replace("T", " "),
                            "Review": h["review"],
                            "Sentiment": h["label"],
                            "Score": h["score"],
                            "Theme": h["theme"],
                            "Urgency": h.get("urgency", "low"),
                            "Root Cause": h.get("root_cause") or "-",
                            "Source": h.get("source", "llm"),
                        }
                        for h in history
                    ])
                    st.dataframe(df_hist, use_container_width=True)
                else:
                    st.info("No feedback records match the current filter.")
        except Exception as e:
            st.error(f"Error fetching history: {e}")
    else:
        st.warning("Backend is currently offline.")
