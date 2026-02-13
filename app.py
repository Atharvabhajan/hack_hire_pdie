import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ───────────────────────────────────────────────────────────────────────────────
# Page config
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pre-Delinquency Intervention Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ───────────────────────────────────────────────────────────────────────────────
# Constants  (DO NOT CHANGE — scoring logic & thresholds)
# ───────────────────────────────────────────────────────────────────────────────
FEATURES = [
    "salary_delay_days",
    "savings_drop_pct",
    "discretionary_spend_change_pct",
    "utility_payment_delay_days",
    "lending_app_upi_txn_count",
    "atm_withdrawal_spike_pct",
    "failed_autodebit",
]

WEIGHTS = {
    "salary_delay_days": 0.18,
    "savings_drop_pct": 0.17,
    "discretionary_spend_change_pct": 0.14,
    "utility_payment_delay_days": 0.14,
    "lending_app_upi_txn_count": 0.12,
    "atm_withdrawal_spike_pct": 0.10,
    "failed_autodebit": 0.15,
}

FEATURE_LABELS = {
    "salary_delay_days": "Salary Delay Days",
    "savings_drop_pct": "Savings Drop %",
    "discretionary_spend_change_pct": "Discretionary Spend Change %",
    "utility_payment_delay_days": "Utility Payment Delay Days",
    "lending_app_upi_txn_count": "Lending-App UPI Txn Count",
    "atm_withdrawal_spike_pct": "ATM Withdrawal Spike %",
    "failed_autodebit": "Failed Autodebit",
}

TIER_COLORS = {
    "Low":    {"bg": "#d1fae5", "fg": "#045e3d", "border": "#6ee7b7", "accent": "#10b981"},
    "Medium": {"bg": "#fed7aa", "fg": "#92400e", "border": "#fb923c", "accent": "#f97316"},
    "High":   {"bg": "#fee2e2", "fg": "#7f1d1d", "border": "#fca5a5", "accent": "#dc2626"},
}


# ───────────────────────────────────────────────────────────────────────────────
# Risk-scoring helpers  (DO NOT CHANGE)
# ───────────────────────────────────────────────────────────────────────────────
def _clip(val, lo, hi):
    return max(lo, min(hi, val))


def _tier(score: float) -> str:
    if score < 0.4:
        return "Low"
    if score <= 0.7:
        return "Medium"
    return "High"


def _risk_components(df: pd.DataFrame) -> pd.DataFrame:
    comp = pd.DataFrame(index=df.index)
    comp["salary_delay_days"] = np.clip((df["salary_delay_days"] - 3) / 4, 0, 1) * WEIGHTS["salary_delay_days"]
    comp["savings_drop_pct"] = np.clip((df["savings_drop_pct"] - 20) / 20, 0, 1) * WEIGHTS["savings_drop_pct"]
    comp["discretionary_spend_change_pct"] = np.clip((-40 - df["discretionary_spend_change_pct"]) / 30, 0, 1) * WEIGHTS[
        "discretionary_spend_change_pct"
    ]
    comp["utility_payment_delay_days"] = np.clip((df["utility_payment_delay_days"] - 5) / 5, 0, 1) * WEIGHTS[
        "utility_payment_delay_days"
    ]
    comp["lending_app_upi_txn_count"] = np.clip((df["lending_app_upi_txn_count"] - 3) / 5, 0, 1) * WEIGHTS[
        "lending_app_upi_txn_count"
    ]
    comp["atm_withdrawal_spike_pct"] = np.clip((df["atm_withdrawal_spike_pct"] - 30) / 30, 0, 1) * WEIGHTS[
        "atm_withdrawal_spike_pct"
    ]
    comp["failed_autodebit"] = df["failed_autodebit"].astype(float) * WEIGHTS["failed_autodebit"]
    return comp


# ───────────────────────────────────────────────────────────────────────────────
# Data generation  (DO NOT CHANGE)
# ───────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_data(n_customers: int = 200, n_weeks: int = 12, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customers = [f"C{idx:03d}" for idx in range(1, n_customers + 1)]
    stressed_customers = set(rng.choice(customers, size=int(n_customers * 0.15), replace=False))
    rows = []

    for cid in customers:
        stressed = cid in stressed_customers
        base_salary_delay = rng.integers(0, 4)
        base_savings_drop = rng.uniform(-5, 18)
        base_discretionary = rng.uniform(-35, 20)
        base_utility_delay = rng.integers(0, 5)
        base_lending_txns = rng.integers(0, 3)
        base_atm_spike = rng.uniform(0, 25)

        for wk in range(1, n_weeks + 1):
            stress_step = 0
            if stressed and wk >= n_weeks - 3:
                stress_step = wk - (n_weeks - 4)

            salary_delay_days = _clip(
                int(round(base_salary_delay + rng.normal(0, 1.2) + stress_step * rng.uniform(0.8, 1.4))), 0, 7
            )
            savings_drop_pct = _clip(
                base_savings_drop + rng.normal(0, 4) + stress_step * rng.uniform(5.0, 8.0), -5, 40
            )
            discretionary_spend_change_pct = _clip(
                base_discretionary + rng.normal(0, 6) - stress_step * rng.uniform(6.0, 9.0), -70, 30
            )
            utility_payment_delay_days = _clip(
                int(round(base_utility_delay + rng.normal(0, 1.5) + stress_step * rng.uniform(1.0, 1.8))), 0, 10
            )
            lending_app_upi_txn_count = _clip(
                int(round(base_lending_txns + rng.normal(0, 1.0) + stress_step * rng.uniform(0.8, 1.3))), 0, 8
            )
            atm_withdrawal_spike_pct = _clip(
                base_atm_spike + rng.normal(0, 7) + stress_step * rng.uniform(6.0, 9.0), 0, 60
            )

            fail_prob = 0.06 + (0.09 * stress_step if stressed else 0)
            failed_autodebit = int(rng.random() < min(0.9, fail_prob))

            rows.append(
                {
                    "customer_id": cid,
                    "week": wk,
                    "salary_delay_days": salary_delay_days,
                    "savings_drop_pct": round(savings_drop_pct, 2),
                    "discretionary_spend_change_pct": round(discretionary_spend_change_pct, 2),
                    "utility_payment_delay_days": utility_payment_delay_days,
                    "lending_app_upi_txn_count": lending_app_upi_txn_count,
                    "atm_withdrawal_spike_pct": round(atm_withdrawal_spike_pct, 2),
                    "failed_autodebit": failed_autodebit,
                }
            )

    df = pd.DataFrame(rows)
    comp = _risk_components(df)
    df["risk_score"] = comp.sum(axis=1).clip(0, 1)
    df["risk_tier"] = df["risk_score"].apply(_tier)

    for feature in FEATURES:
        df[f"contrib_{feature}"] = comp[feature]

    return df


# ───────────────────────────────────────────────────────────────────────────────
# Analytical helpers  (DO NOT CHANGE)
# ───────────────────────────────────────────────────────────────────────────────
def top_drivers(row: pd.Series, n: int = 3):
    pairs = [(feature, float(row[f"contrib_{feature}"])) for feature in FEATURES]
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    return pairs[:n]


def feature_explanation(feature: str, row: pd.Series) -> str:
    if feature == "salary_delay_days":
        return f"Salary credit was delayed by {int(row[feature])} days, indicating cashflow stress before EMI dates."
    if feature == "savings_drop_pct":
        return f"Savings dropped by {row[feature]:.1f}%, reducing short-term repayment buffer."
    if feature == "discretionary_spend_change_pct":
        return f"Discretionary spend changed by {row[feature]:.1f}%; sharp cutbacks can signal tightening liquidity."
    if feature == "utility_payment_delay_days":
        return f"Utility payments were delayed by {int(row[feature])} days, often preceding broader payment stress."
    if feature == "lending_app_upi_txn_count":
        return f"{int(row[feature])} lending-app UPI transactions suggest increased short-term borrowing behavior."
    if feature == "atm_withdrawal_spike_pct":
        return f"ATM withdrawals spiked by {row[feature]:.1f}%, which may indicate emergency cash dependence."
    return "Failed autodebit was observed, directly indicating repayment friction."


def _risk_trend_label(group: pd.DataFrame) -> str:
    recent = group.sort_values("week").tail(3)
    slope = float(np.polyfit(recent["week"], recent["risk_score"], 1)[0])
    if slope > 0.015:
        return "Rising"
    if slope < -0.015:
        return "Falling"
    return "Stable"


@st.cache_data(show_spinner=False)
def latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    latest_week = int(df["week"].max())
    latest = df[df["week"] == latest_week].copy()
    trends = df.groupby("customer_id", group_keys=False).apply(_risk_trend_label).rename("risk_trend").reset_index()
    latest["top_reason"] = latest.apply(lambda row: FEATURE_LABELS[top_drivers(row, 1)[0][0]], axis=1)
    latest = latest[["customer_id", "risk_score", "risk_tier", "top_reason"]].merge(trends, on="customer_id", how="left")
    latest["risk_score"] = latest["risk_score"].round(3)
    latest = latest[["customer_id", "risk_score", "risk_tier", "risk_trend", "top_reason"]]
    latest = latest.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return latest


def intervention_engine(row: pd.Series, driver_labels: list[str]):
    if row["salary_delay_days"] > 3 and row["savings_drop_pct"] > 20:
        action = "Offer EMI date shift / short grace period"
        message = "We noticed temporary cashflow pressure. We can shift your EMI date or provide a short grace period to avoid penalties."
    elif row["failed_autodebit"] == 1:
        action = "Immediate proactive outreach + payment retry scheduling"
        message = "Your last autodebit did not go through. We can help schedule a retry at your preferred date and confirm account setup."
    elif row["lending_app_upi_txn_count"] >= 3:
        action = "Offer restructuring / lower-cost credit line"
        message = "We can review your repayment plan and provide a lower-cost structured option to reduce monthly stress."
    elif row["risk_tier"] == "Medium":
        action = "Soft nudge + budgeting tips"
        message = "You are still in control. A quick budget tune-up and payment reminder can help keep your account healthy."
    elif row["risk_tier"] == "High":
        action = "Priority relationship-manager outreach with repayment planning"
        message = "A relationship manager can contact you today to set up a sustainable repayment plan and avoid delinquency."
    else:
        action = "Continue monitoring"
        message = "No urgent intervention needed right now. We will continue monitoring weekly patterns."

    rationale = f"Recommended because key drivers this week are {', '.join(driver_labels[:2])}."
    return action, message, rationale


@st.cache_data(show_spinner=False)
def build_hover_drivers(cust_df: pd.DataFrame) -> pd.DataFrame:
    out = cust_df.copy()
    top2 = out.apply(lambda row: top_drivers(row, 2), axis=1)
    out["driver_1"] = top2.apply(lambda arr: FEATURE_LABELS[arr[0][0]])
    out["driver_2"] = top2.apply(lambda arr: FEATURE_LABELS[arr[1][0]])
    return out


# ───────────────────────────────────────────────────────────────────────────────
# Theme & CSS
# ───────────────────────────────────────────────────────────────────────────────
def apply_theme():
    st.markdown(
        """
        <style>
        /* ── Global ──────────────────────────────────────────── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        .stApp {
            background: #f8f9fc;
            color: #1e293b;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }

        /* ── Hero Header ─────────────────────────────────────── */
        .hero {
            background: #0f172a;
            color: #ffffff;
            padding: 28px 36px 22px;
            border-radius: 14px;
            margin-bottom: 24px;
            border-bottom: 3px solid #0d9488;
        }
        .hero h1 {
            margin: 0 0 6px 0;
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }
        .hero .tagline {
            font-size: 0.9rem;
            color: #94a3b8;
            font-weight: 400;
            line-height: 1.6;
            max-width: 720px;
        }
        .hero .badge-row {
            margin-top: 14px;
            display: flex;
            gap: 10px;
        }
        .hero .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
            padding: 5px 14px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 500;
            color: #cbd5e1;
            letter-spacing: 0.01em;
        }

        /* ── Cards ───────────────────────────────────────────── */
        .card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 20px 22px;
            box-shadow: 0 1px 3px rgba(15,23,42,0.04);
        }
        .card-title {
            font-weight: 700;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748b;
            margin-bottom: 12px;
        }

        /* ── KPI metric cards ────────────────────────────────── */
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 18px 20px 16px;
            box-shadow: 0 1px 3px rgba(15,23,42,0.04);
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.74rem !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #475569 !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 800 !important;
            color: #0f172a !important;
            letter-spacing: -0.01em;
        }

        /* ── Colored top-border KPI variants ─────────────────── */
        .kpi-high [data-testid="stMetric"] { border-top: 4px solid #dc2626; background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); }
        .kpi-medium [data-testid="stMetric"] { border-top: 4px solid #f97316; background: linear-gradient(135deg, #ffedd5 0%, #fed7aa 100%); }
        .kpi-ok [data-testid="stMetric"] { border-top: 4px solid #10b981; background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); }
        .kpi-neutral [data-testid="stMetric"] { border-top: 4px solid #2563eb; background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); }

        /* ── Pills / Badges ──────────────────────────────────── */
        .pill {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 999px;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            border: 1px solid transparent;
            vertical-align: middle;
        }
        .pill-low    { background:#d1fae5; color:#045e3d; border-color:#6ee7b7; font-weight:700; }
        .pill-medium { background:#fed7aa; color:#92400e; border-color:#fb923c; font-weight:700; }
        .pill-high   { background:#fee2e2; color:#7f1d1d; border-color:#fca5a5; font-weight:700; }

        /* ── Legend card ──────────────────────────────────────── */
        .legend-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 6px;
            font-size: 0.86rem;
            color: #475569;
        }
        .legend-row .action-text {
            color: #334155;
            font-weight: 500;
        }

        /* ── Tabs ─────────────────────────────────────────────── */
        button[data-baseweb="tab"] {
            font-weight: 700 !important;
            font-size: 0.9rem !important;
            padding: 12px 24px !important;
            letter-spacing: 0.01em;
        }

        /* ── Intervention card ───────────────────────────────── */
        .intervention-card {
            background: linear-gradient(135deg, #f0fdf4 0%, #f8fffe 100%);
            border: 2px solid #10b981;
            border-left: 5px solid #059669;
            border-radius: 14px;
            padding: 24px 28px;
            box-shadow: 0 4px 12px rgba(16,185,129,0.12);
        }
        .intervention-card .field-label {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #059669;
            margin-bottom: 3px;
        }
        .intervention-card .field-value {
            font-size: 0.94rem;
            color: #1e293b;
            margin-bottom: 16px;
            line-height: 1.6;
            font-weight: 500;
        }
        .intervention-card .channel-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            border: 2px solid #6ee7b7;
            padding: 8px 18px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 700;
            color: #045e3d;
        }

        /* ── Driver card ─────────────────────────────────────── */
        .driver-card {
            background: #ffffff;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(15,23,42,0.06);
            transition: all 0.2s ease;
        }
        .driver-card:hover {
            border-color: #10b981;
            box-shadow: 0 4px 12px rgba(16,185,129,0.15);
        }
        .progress-bar-bg {
            background: #f3f4f6;
            border-radius: 999px;
            height: 8px;
            width: 100%;
            margin-top: 10px;
            overflow: hidden;
            border: 1px solid #e5e7eb;
        }
        .progress-bar-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 0.4s ease;
        }

        /* ── Gauge ────────────────────────────────────────────── */
        .gauge-container {
            text-align: center;
            padding: 10px 0;
        }

        /* ── Sidebar ─────────────────────────────────────────── */
        section[data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid #e2e8f0;
        }
        section[data-testid="stSidebar"] .stSelectbox label {
            font-weight: 700;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #475569;
        }

        /* ── How-it-works step cards ─────────────────────────── */
        .step-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 22px 24px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(15,23,42,0.04);
            height: 100%;
        }
        .step-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 34px;
            height: 34px;
            background: #0f172a;
            color: #ffffff;
            border-radius: 50%;
            font-weight: 800;
            font-size: 0.85rem;
            margin-bottom: 12px;
        }
        .step-card h4 {
            margin: 0 0 6px 0;
            font-size: 0.95rem;
            font-weight: 700;
            color: #0f172a;
        }
        .step-card p {
            margin: 0;
            font-size: 0.82rem;
            color: #64748b;
            line-height: 1.55;
        }

        /* ── Impact Simulation ────────────────────────────────── */
        .impact-group {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: 22px 24px;
            box-shadow: 0 1px 3px rgba(15,23,42,0.04);
        }
        .impact-group-title {
            font-weight: 700;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748b;
            margin-bottom: 14px;
            padding-bottom: 8px;
            border-bottom: 1px solid #f1f5f9;
        }
        .impact-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            padding: 7px 0;
            font-size: 0.88rem;
            color: #334155;
            border-bottom: 1px solid #f8fafc;
        }
        .impact-row:last-child { border-bottom: none; }
        .impact-row .label { color: #64748b; font-weight: 500; }
        .impact-row .value { font-weight: 700; color: #0f172a; font-size: 0.95rem; }
        .impact-highlight {
            background: linear-gradient(135deg, #d1fae5 0%, #f0fdf4 100%);
            border: 2px solid #10b981;
            border-left: 5px solid #059669;
        }
        .impact-highlight .impact-group-title { color: #065f46; font-weight: 700; }
        .impact-highlight .value { color: #047857; font-weight: 700; }
        .impact-saved {
            display: inline-block;
            background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
            color: #045e3d;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 8px;
            font-size: 0.95rem;
            border: 1px solid #6ee7b7;
        }

        /* ── Misc ─────────────────────────────────────────────── */
        .note {
            color: #94a3b8;
            font-size: 0.8rem;
            margin-top: 6px;
        }
        .section-desc {
            color: #64748b;
            font-size: 0.88rem;
            margin-bottom: 16px;
            line-height: 1.55;
        }
        .footer {
            text-align: center;
            color: #94a3b8;
            font-size: 0.76rem;
            margin-top: 40px;
            padding: 20px 0;
            border-top: 1px solid #e2e8f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ───────────────────────────────────────────────────────────────────────────────
# UI render helpers
# ───────────────────────────────────────────────────────────────────────────────
def _pill_html(tier: str) -> str:
    cls = {"Low": "pill-low", "Medium": "pill-medium", "High": "pill-high"}.get(tier, "")
    return f'<span class="pill {cls}">{tier}</span>'


def _fmt_inr(value: float) -> str:
    """Format a number as Indian Rupee string."""
    if value >= 1e7:
        return f"\u20B9{value / 1e7:,.2f} Cr"
    if value >= 1e5:
        return f"\u20B9{value / 1e5:,.2f} L"
    return f"\u20B9{value:,.0f}"


def render_header():
    st.markdown(
        """
        <div class="hero">
            <div style="display:flex; align-items:center; gap:20px; margin-bottom:12px;">
                <div style="font-size:3.2rem;">🛡️</div>
                <h1 style="margin:0;">Pre-Delinquency Intervention Engine</h1>
            </div>
            <div class="tagline">
                An early-warning system that monitors 7 behavioural signals across the retail loan portfolio
                every week &mdash; flagging customers showing signs of financial stress <b>before</b> they miss a payment,
                and recommending personalised interventions to prevent delinquency.
            </div>
            <div class="badge-row">
                <span class="hero-badge">200 Customers</span>
                <span class="hero-badge">12 Weeks Tracked</span>
                <span class="hero-badge">7 Behavioural Signals</span>
                <span class="hero-badge">Weekly Scoring</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(latest: pd.DataFrame):
    high_count = int((latest["risk_tier"] == "High").sum())
    medium_count = int((latest["risk_tier"] == "Medium").sum())
    avg_risk = float(latest["risk_score"].mean())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown('<div class="kpi-neutral">', unsafe_allow_html=True)
        st.metric("\u2713 Monitored", f"{len(latest):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="kpi-high">', unsafe_allow_html=True)
        st.metric("\u26A0 High Risk", high_count)
        st.markdown('</div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="kpi-medium">', unsafe_allow_html=True)
        st.metric("\u25CF Medium Risk", medium_count)
        st.markdown('</div>', unsafe_allow_html=True)
    with k4:
        st.markdown('<div class="kpi-ok">', unsafe_allow_html=True)
        st.metric("\u2192 Avg Risk", f"{avg_risk:.3f}")
        st.markdown('</div>', unsafe_allow_html=True)


def render_legend():
    st.markdown(
        """
        <div class="card" style="margin-top:8px; margin-bottom:24px;">
            <div class="card-title">Risk Tier Classification</div>
            <div class="legend-row">
                <span class="pill pill-low">Low</span>
                <span>Score &lt; 0.40</span>
                <span style="color:#94a3b8;">&#x2192;</span>
                <span class="action-text">Continue monitoring &mdash; no action needed</span>
            </div>
            <div class="legend-row">
                <span class="pill pill-medium">Medium</span>
                <span>Score 0.40 &ndash; 0.70</span>
                <span style="color:#94a3b8;">&#x2192;</span>
                <span class="action-text">Preventive outreach &mdash; soft nudge &amp; budgeting support</span>
            </div>
            <div class="legend-row">
                <span class="pill pill-high">High</span>
                <span>Score &gt; 0.70</span>
                <span style="color:#94a3b8;">&#x2192;</span>
                <span class="action-text">Immediate intervention &mdash; restructuring or RM outreach</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _risk_tier_style(val: str) -> str:
    colors = TIER_COLORS.get(val)
    if colors:
        return f"background-color:{colors['bg']}; color:{colors['fg']}; font-weight:600; border-radius:6px;"
    return ""


def _make_gauge(score: float, tier: str) -> go.Figure:
    color = TIER_COLORS[tier]["accent"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 3),
        number={"font": {"size": 38, "color": "#0f172a", "family": "Inter"}, "valueformat": ".3f"},
        gauge={
            "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#e2e8f0",
                     "tickfont": {"size": 10, "color": "#94a3b8"}},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "#f1f5f9",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 0.4], "color": "#ecfdf3"},
                {"range": [0.4, 0.7], "color": "#fff7ed"},
                {"range": [0.7, 1], "color": "#fef2f2"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": round(score, 3),
            },
        },
    ))
    fig.update_layout(
        height=190,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
    )
    return fig


def _make_donut(latest: pd.DataFrame) -> go.Figure:
    counts = latest["risk_tier"].value_counts()
    labels = ["High", "Medium", "Low"]
    values = [int(counts.get(l, 0)) for l in labels]
    colors = ["#ef4444", "#f59e0b", "#22c55e"]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=3)),
        textinfo="label+value",
        textfont=dict(size=12, family="Inter"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        showlegend=False,
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        annotations=[dict(text="Risk<br>Split", x=0.5, y=0.5, font_size=13,
                          font_color="#64748b", showarrow=False, font_family="Inter")],
    )
    return fig


# ───────────────────────────────────────────────────────────────────────────────
# Portfolio tab
# ───────────────────────────────────────────────────────────────────────────────
def render_portfolio_tab(df: pd.DataFrame, latest: pd.DataFrame):
    st.markdown(
        '<p class="section-desc">'
        'Snapshot of all <b>200 customers</b> for the most recent monitoring week. '
        'The table is sorted by risk score (highest first). '
        'Coloured tiers indicate urgency of outreach.'
        '</p>',
        unsafe_allow_html=True,
    )

    # ── Distribution chart + table side by side ──
    col_chart, col_table = st.columns([1, 2.8])

    with col_chart:
        st.markdown(
            '<div class="card" style="padding:16px 12px;">'
            '<div class="card-title">Tier Distribution</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_make_donut(latest), use_container_width=True)

        # Trend summary
        rising = int((latest["risk_trend"] == "Rising").sum())
        falling = int((latest["risk_trend"] == "Falling").sum())
        stable = int((latest["risk_trend"] == "Stable").sum())
        st.markdown(
            f'<div class="card" style="margin-top:12px;">'
            f'<div class="card-title">Trend Summary (3-Week Slope)</div>'
            f'<div style="font-size:0.88rem; color:#475569; line-height:1.8;">'
            f'Rising: <b>{rising}</b><br>'
            f'Stable: <b>{stable}</b><br>'
            f'Falling: <b>{falling}</b>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with col_table:
        overview = latest[["customer_id", "risk_score", "risk_tier", "risk_trend", "top_reason"]].copy()
        # Add trend symbols for visual clarity
        overview["Trend"] = latest["risk_trend"].apply(lambda x: "\u2191" if x == "Rising" else "\u2193" if x == "Falling" else "\u2192")
        overview = overview.drop(columns=["risk_trend"]) if "risk_trend" in latest.columns else overview
        overview.columns = ["Customer ID", "Risk Score", "Risk Tier", "Trend Symbol", "Top Reason"]
        styled = (
            overview.style
            .format({"Risk Score": "{:.3f}"})
            .bar(subset=["Risk Score"], color="#d1fae5", vmin=0, vmax=1)
            .map(_risk_tier_style, subset=["Risk Tier"])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True, height=520)

    st.markdown("---")

    # ── Case Queue (High Risk) ──
    st.markdown("##### ► Generated Case Queue (High Risk)")
    st.markdown(
        '<p class="section-desc">'  
        'Actionable case list for the collections / RM team. '
        'Priority is assigned as P1 (High + Rising) or P2 (High + Stable/Falling). '
        'Recommended actions are generated by the intervention engine.'
        '</p>',
        unsafe_allow_html=True,
    )

    high_latest = latest[latest["risk_tier"] == "High"].copy()
    if high_latest.empty:
        st.success("No customers are currently in the High risk tier. Portfolio is healthy.")
    else:
        latest_week = int(df["week"].max())
        high_ids = high_latest["customer_id"].tolist()
        raw_high = df[(df["week"] == latest_week) & (df["customer_id"].isin(high_ids))]

        queue_rows = []
        for _, snap_row in high_latest.iterrows():
            cid = snap_row["customer_id"]
            raw_row = raw_high[raw_high["customer_id"] == cid].iloc[0]
            drivers = top_drivers(raw_row, 2)
            driver_labels = [FEATURE_LABELS[d[0]] for d in drivers]
            action, _, _ = intervention_engine(raw_row, driver_labels)
            channel = "Phone + In-app"
            priority = "P1" if snap_row["risk_trend"] == "Rising" else "P2"
            queue_rows.append({
                "Customer ID": cid,
                "Risk Score": snap_row["risk_score"],
                "Tier": snap_row["risk_tier"],
                "Trend": snap_row["risk_trend"],
                "Top Reason": snap_row["top_reason"],
                "Recommended Action": action,
                "Channel": channel,
                "Priority": priority,
            })

        queue_df = pd.DataFrame(queue_rows)
        queue_df = queue_df.sort_values(["Priority", "Risk Score"], ascending=[True, False]).reset_index(drop=True)
        st.dataframe(
            queue_df.style
            .format({"Risk Score": "{:.3f}"})
            .map(_risk_tier_style, subset=["Tier"]),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        '<p class="note">Select a customer in the sidebar, then switch to the '
        '<b>Customer View</b> tab for the full risk profile and intervention details.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Operationalization note ──
    st.markdown("##### ⚙ Operationalization")
    st.markdown(
        '<div class="card">'
        '<div style="font-size:0.86rem; color:#475569; line-height:1.7;">'
        '<b>How this integrates into bank operations:</b>'
        '<ul style="margin:8px 0 0 0; padding-left:20px;">'
        '<li>Dashboard refreshed on a weekly cycle; can be scheduled for daily runs with live transaction feeds.</li>'
        '<li>Case queue exported to the RM / collections CRM for assignment, tracking, and SLA monitoring.</li>'
        '<li>Contact outcomes (connected, promise-to-pay, restructured) fed back for weight recalibration and model tuning (future roadmap).</li>'
        '</ul>'
        '</div></div>',
        unsafe_allow_html=True,
    )


# ───────────────────────────────────────────────────────────────────────────────
# Customer View tab
# ───────────────────────────────────────────────────────────────────────────────
def render_customer_tab(df: pd.DataFrame, selected_customer: str, selected_week: int):
    cust_df = df[df["customer_id"] == selected_customer].sort_values("week").copy()
    selected_row = cust_df[cust_df["week"] == selected_week].iloc[0]
    selected_top3 = top_drivers(selected_row, 3)
    selected_driver_labels = [FEATURE_LABELS[item[0]] for item in selected_top3]
    tier = selected_row["risk_tier"]
    score = float(selected_row["risk_score"])

    # ── Risk stability (last 4 weeks) ──
    last4 = cust_df.sort_values("week").tail(4)["risk_score"]
    risk_std = float(last4.std()) if len(last4) >= 2 else 0.0
    if risk_std < 0.05:
        stability_label, stability_cls = "High stability", "pill-low"
    elif risk_std <= 0.12:
        stability_label, stability_cls = "Medium stability", "pill-medium"
    else:
        stability_label, stability_cls = "Low stability", "pill-high"

    # ── Lead time indicator ──
    crossed_this_week = False
    if score > 0.7:
        prev_weeks = cust_df[cust_df["week"] < selected_week].sort_values("week")
        if not prev_weeks.empty and float(prev_weeks.iloc[-1]["risk_score"]) <= 0.7:
            crossed_this_week = True
    lead_time = "3 weeks (simulated)" if crossed_this_week else "2\u20134 weeks (simulated)"

    # ── Header row: Customer ID + Gauge ──
    h1, h2 = st.columns([2.2, 1])
    with h1:
        st.markdown(
            f'<div style="display:flex; align-items:center; gap:14px; margin-bottom:4px;">'
            f'<span style="font-size:1.3rem; font-weight:800; color:#0f172a;">{selected_customer}</span>'
            f'{_pill_html(tier)}'
            f'<span class="pill {stability_cls}" style="font-size:0.68rem;">{stability_label}</span>'
            f'<span style="color:#94a3b8; font-size:0.85rem;">Week {selected_week}</span>'
            f'</div>'
            f'<p class="section-desc" style="margin-top:4px;">'
            f'Detailed risk profile and behavioural signals for this customer. '
            f'Prediction horizon: <b>{lead_time}</b> before EMI miss (simulated).'
            f'</p>',
            unsafe_allow_html=True,
        )
    with h2:
        st.plotly_chart(_make_gauge(score, tier), use_container_width=True)
        st.markdown(
            f'<p style="text-align:center; font-size:0.76rem; color:#94a3b8; margin-top:-8px;">'
            f'Stability: {stability_label} (4-wk std: {risk_std:.3f})<br>'
            f'Stable upward risk is prioritised for interventions.</p>',
            unsafe_allow_html=True,
        )

    # ── Snapshot metrics ──
    st.markdown(
        '<div class="card" style="margin-bottom:4px; padding:12px 18px;">'
        '<div class="card-title">Signal Snapshot &mdash; Week ' + str(selected_week) + '</div></div>',
        unsafe_allow_html=True,
    )
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Salary Delay", f"{int(selected_row['salary_delay_days'])} days")
    s2.metric("Savings Drop", f"{selected_row['savings_drop_pct']:.1f}%")
    s3.metric("Discr. Spend", f"{selected_row['discretionary_spend_change_pct']:.1f}%")
    s4.metric("Utility Delay", f"{int(selected_row['utility_payment_delay_days'])} days")

    s5, s6, s7, _ = st.columns(4)
    s5.metric("Lending-App Txns", int(selected_row["lending_app_upi_txn_count"]))
    s6.metric("ATM Spike", f"{selected_row['atm_withdrawal_spike_pct']:.1f}%")
    s7.metric("Failed Autodebit", "Yes" if int(selected_row["failed_autodebit"]) else "No")

    st.markdown("")

    # ── Risk Trend + Why Flagged ──
    c1, c2 = st.columns([1.6, 1])

    with c1:
        st.markdown("##### ▼ Risk Score Trend")
        st.markdown(
            '<p class="section-desc" style="margin-bottom:8px;">12-week trajectory. '
            'Hover over points to see the top contributing signals at each week.</p>',
            unsafe_allow_html=True,
        )
        hover_df = build_hover_drivers(cust_df)
        fig = go.Figure()

        # Area fill under curve
        fig.add_trace(go.Scatter(
            x=hover_df["week"], y=hover_df["risk_score"],
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(13,148,136,0.06)",
            showlegend=False,
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=hover_df["week"],
            y=hover_df["risk_score"],
            mode="lines+markers",
            line=dict(color="#0d9488", width=3, shape="spline"),
            marker=dict(size=8, color="#0d9488", line=dict(width=2, color="#ffffff")),
            customdata=np.stack([hover_df["driver_1"], hover_df["driver_2"]], axis=-1),
            hovertemplate=(
                "<b>Week %{x}</b><br>"
                "Risk Score: %{y:.3f}<br>"
                "Driver 1: %{customdata[0]}<br>"
                "Driver 2: %{customdata[1]}<extra></extra>"
            ),
            showlegend=False,
        ))

        fig.add_hline(y=0.4, line_dash="dot", line_color="#f59e0b", line_width=1.5,
                      annotation_text="Medium (0.4)", annotation_font_size=11, annotation_font_color="#f59e0b")
        fig.add_hline(y=0.7, line_dash="dot", line_color="#ef4444", line_width=1.5,
                      annotation_text="High (0.7)", annotation_font_size=11, annotation_font_color="#ef4444")

        cross_week = None
        cross_score = None
        for i in range(len(hover_df)):
            sc = float(hover_df.iloc[i]["risk_score"])
            prev = float(hover_df.iloc[i - 1]["risk_score"]) if i > 0 else 0.0
            if sc > 0.7 and prev <= 0.7:
                cross_week = int(hover_df.iloc[i]["week"])
                cross_score = sc
                break

        if cross_week is not None:
            fig.add_annotation(
                x=cross_week, y=cross_score,
                text="High-risk trigger",
                showarrow=True, arrowhead=2, arrowcolor="#ef4444",
                bgcolor="#fef2f2", bordercolor="#fecaca", borderwidth=1,
                font=dict(size=11, color="#991b1b"),
            )

        fig.update_layout(
            xaxis_title="Week", yaxis_title="Risk Score",
            yaxis=dict(range=[0, 1]),
            plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#f1f5f9", dtick=1),
            yaxis_gridcolor="#f1f5f9",
            font=dict(family="Inter, sans-serif", size=12),
            hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", font_size=12),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### ★ Top Risk Drivers")
        st.markdown(
            '<p class="section-desc" style="margin-bottom:8px;">'  
            'Top 3 risk drivers and their contribution to the composite score.</p>',
            unsafe_allow_html=True,
        )
        total_risk = float(selected_row["risk_score"])
        for feature, contribution in selected_top3:
            pct = (contribution / total_risk * 100) if total_risk > 0 else 0
            label = FEATURE_LABELS[feature]
            explanation = feature_explanation(feature, selected_row)
            bar_color = "#ef4444" if pct > 30 else "#f59e0b" if pct > 15 else "#22c55e"
            pill_cls = "pill-high" if pct > 30 else "pill-medium" if pct > 15 else "pill-low"
            st.markdown(
                f'<div class="driver-card">'
                f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                f'<span style="font-weight:700; font-size:0.88rem; color:#1e293b;">{label}</span>'
                f'<span class="pill {pill_cls}" style="font-size:0.72rem;">{pct:.0f}%</span>'
                f'</div>'
                f'<div class="progress-bar-bg">'
                f'<div class="progress-bar-fill" style="width:{min(pct, 100):.0f}%; background:{bar_color};"></div>'
                f'</div>'
                f'<div style="font-size:0.8rem; color:#64748b; line-height:1.5; margin-top:8px;">{explanation}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Weekly signals ──
    with st.expander("Weekly Signal History (all 12 weeks)"):
        display_cols = [
            "week", "salary_delay_days", "savings_drop_pct",
            "discretionary_spend_change_pct", "utility_payment_delay_days",
            "lending_app_upi_txn_count", "atm_withdrawal_spike_pct",
            "failed_autodebit", "risk_score", "risk_tier",
        ]
        display_df = cust_df[display_cols].copy()
        display_df["risk_score"] = display_df["risk_score"].round(3)
        display_df.columns = [
            "Week", "Salary Delay", "Savings Drop %", "Discr. Spend %",
            "Utility Delay", "Lending Txns", "ATM Spike %", "Failed AD",
            "Risk Score", "Tier",
        ]
        st.dataframe(
            display_df.style.map(_risk_tier_style, subset=["Tier"]),
            use_container_width=True,
            hide_index=True,
        )

    # ── Intervention ──
    action, message, rationale = intervention_engine(selected_row, selected_driver_labels)
    if tier == "High":
        channel = "Phone call + In-app notification"
    elif tier == "Medium":
        channel = "In-app notification + SMS"
    else:
        channel = "Monitor only"

    st.markdown("##### ✎ Recommended Intervention")
    st.markdown(
        f"""
        <div class="intervention-card">
            <div class="field-label">Recommended Action</div>
            <div class="field-value" style="font-weight:600;">{action}</div>
            <div class="field-label">Suggested Customer Message</div>
            <div class="field-value" style="font-style:italic; border-left:3px solid #e2e8f0; padding-left:14px;">
                &ldquo;{message}&rdquo;
            </div>
            <div class="field-label">Rationale</div>
            <div class="field-value">{rationale}</div>
            <div class="field-label">Next Best Channel</div>
            <div class="field-value" style="margin-bottom:0;">
                <span class="channel-badge">{channel}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Prevention Strategy Logic ──
    st.markdown("##### Prevention Strategy Logic")
    st.markdown(
        '<p class="section-desc">'
        'How detected signals map to preventive actions designed to reduce delinquency risk.'
        '</p>',
        unsafe_allow_html=True,
    )

    signal_to_mechanism = {
        "salary_delay_days": "EMI date realignment / grace period",
        "savings_drop_pct": "Temporary liquidity support / short deferment",
        "discretionary_spend_change_pct": "Budget advisory + soft reminder",
        "utility_payment_delay_days": "Payment prioritization alert",
        "lending_app_upi_txn_count": "Offer lower-cost internal credit",
        "atm_withdrawal_spike_pct": "Liquidity stress outreach",
        "failed_autodebit": "Mandate re-authorization flow",
    }

    strategy_rows = []
    for feature, contribution in selected_top3:
        mechanism = signal_to_mechanism.get(feature, "Custom intervention")
        strategy_rows.append({
            "Triggered Signal": FEATURE_LABELS[feature],
            "Preventive Mechanism": mechanism,
        })

    strategy_df = pd.DataFrame(strategy_rows)
    st.dataframe(
        strategy_df.style
        .set_properties(**{"text-align": "left"})
        .set_properties(**{"padding": "12px"}),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("")

    # ── Estimated Impact After Intervention ──
    st.markdown("##### Estimated Impact (Scenario-Based Projection)")
    st.markdown(
        '<p class="section-desc">'
        'Scenario-based risk adjustment contingent on intervention acceptance and execution effectiveness.'
        '</p>',
        unsafe_allow_html=True,
    )

    reduction_factor = 0.0
    if tier == "High":
        reduction_factor = 0.40
    elif tier == "Medium":
        reduction_factor = 0.20
    else:
        reduction_factor = 0.05

    adjusted_score = score * (1 - reduction_factor)
    adjusted_tier = _tier(adjusted_score)

    st.markdown(
        '<div style="font-size:0.82rem; color:#64748b; margin-bottom:16px; font-style:italic;"'
        '>Assumed intervention acceptance rate: 60%</div>',
        unsafe_allow_html=True,
    )

    before_col, after_col = st.columns(2)

    with before_col:
        st.markdown(
            f'<div class="card" style="background:linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);">'
            f'<div class="card-title">Before Intervention</div>'
            f'<div style="font-size:0.88rem; color:#7f1d1d; line-height:1.8; margin-top:12px;">'
            f'Risk Score: <b style="font-size:1.2rem;">{score:.3f}</b><br>'
            f'Risk Tier: <span class="pill pill-{tier.lower()}" style="margin-top:8px;">{tier}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    with after_col:
        st.markdown(
            f'<div class="card" style="background:linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);">'
            f'<div class="card-title">Expected Post-Intervention Risk (If Accepted)</div>'
            f'<div style="font-size:0.88rem; color:#045e3d; line-height:1.8; margin-top:12px;">'
            f'Risk Score: <b style="font-size:1.2rem;">{adjusted_score:.3f}</b><br>'
            f'Risk Tier: <span class="pill pill-{adjusted_tier.lower()}" style="margin-top:8px;">{adjusted_tier}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p class="note" style="margin-top:16px; padding:0;">'
        'Scenario projection based on assumed intervention effectiveness and customer acceptance probability. '
        'Actual outcomes may vary depending on execution quality and customer response.</p>',
        unsafe_allow_html=True,
    )


# ───────────────────────────────────────────────────────────────────────────────
# Impact Simulation tab
# ───────────────────────────────────────────────────────────────────────────────
def render_impact_tab(latest: pd.DataFrame):
    st.markdown(
        '<p class="section-desc">'
        'Financial projection comparing expected credit losses <b>without</b> early intervention '
        'versus <b>with</b> the engine active. Adjust the assumptions in the sidebar to model different scenarios.'
        '</p>',
        unsafe_allow_html=True,
    )

    # ── Counts ──
    high_count = int((latest["risk_tier"] == "High").sum())
    medium_count = int((latest["risk_tier"] == "Medium").sum())
    eligible_cases = high_count + medium_count

    # ── Sidebar assumptions ──
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<div style="font-weight:700; font-size:0.78rem; text-transform:uppercase; '
            'letter-spacing:0.05em; color:#475569; margin-bottom:8px;">Simulation Parameters</div>',
            unsafe_allow_html=True,
        )
        avg_exposure = st.number_input(
            "Avg. exposure per customer (\u20B9)", min_value=10000, max_value=500000,
            value=50000, step=5000, format="%d",
        )
        baseline_high = st.slider(
            "Baseline default rate \u2014 High risk", 5, 40, 15, 1, format="%d%%",
        ) / 100.0
        baseline_medium = st.slider(
            "Baseline default rate \u2014 Medium risk", 2, 25, 7, 1, format="%d%%",
        ) / 100.0
        collection_cost_pct = st.slider(
            "Collection cost (% of recovered)", 5, 40, 18, 1, format="%d%%",
        ) / 100.0
        eff_high = st.slider(
            "Intervention effect \u2014 High risk (default reduction)", 10, 70, 40, 5, format="%d%%",
        ) / 100.0
        eff_medium = st.slider(
            "Intervention effect \u2014 Medium risk (default reduction)", 5, 50, 20, 5, format="%d%%",
        ) / 100.0

        st.markdown("---")
        st.markdown(
            '<div style="font-weight:700; font-size:0.78rem; text-transform:uppercase; '
            'letter-spacing:0.05em; color:#475569; margin-bottom:8px;">Advanced Economics</div>',
            unsafe_allow_html=True,
        )
        lgd_pct = st.slider(
            "Loss Given Default (LGD)", 20, 90, 55, 5, format="%d%%",
        ) / 100.0
        recovery_rate = st.slider(
            "Baseline recovery (cure) rate", 20, 95, 60, 5, format="%d%%",
        ) / 100.0
        recovery_uplift = st.slider(
            "Recovery uplift with engine", 0, 30, 10, 5, format="+%d%%",
        ) / 100.0
        cost_per_high = st.slider(
            "Outreach cost per High-risk case (\u20B9)", 0, 500, 60, 10,
        )
        cost_per_medium = st.slider(
            "Outreach cost per Medium-risk case (\u20B9)", 0, 200, 10, 5,
        )
        outreach_capacity = st.slider(
            "Weekly outreach capacity (cases)", 20, 500, 120, 10,
        )

    # ── Capacity split ──
    # High-risk customers have higher scores, so they are contacted first
    contacted_high = min(high_count, outreach_capacity)
    remaining_cap = outreach_capacity - contacted_high
    contacted_medium = min(medium_count, max(0, remaining_cap))
    non_contacted_high = high_count - contacted_high
    non_contacted_medium = medium_count - contacted_medium
    overflow = max(0, eligible_cases - outreach_capacity)

    # ── Capacity KPI ──
    cap_cols = st.columns(4)
    with cap_cols[0]:
        st.markdown('<div class="kpi-neutral">', unsafe_allow_html=True)
        st.metric("Eligible Cases", eligible_cases)
        st.markdown('</div>', unsafe_allow_html=True)
    with cap_cols[1]:
        st.markdown('<div class="kpi-ok">', unsafe_allow_html=True)
        st.metric("Contacted This Week", min(eligible_cases, outreach_capacity))
        st.markdown('</div>', unsafe_allow_html=True)
    with cap_cols[2]:
        if overflow > 0:
            st.markdown('<div class="kpi-high">', unsafe_allow_html=True)
            st.metric("Overflow (Not Contacted)", overflow)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="kpi-ok">', unsafe_allow_html=True)
            st.metric("Overflow", "None")
            st.markdown('</div>', unsafe_allow_html=True)
    with cap_cols[3]:
        st.markdown('<div class="kpi-medium">', unsafe_allow_html=True)
        st.metric("Capacity Utilisation", f"{min(eligible_cases, outreach_capacity) / outreach_capacity * 100:.0f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # ── WITHOUT engine ──
    defaults_wo = (high_count * baseline_high) + (medium_count * baseline_medium)
    ead_wo = defaults_wo * avg_exposure
    credit_loss_wo = ead_wo * lgd_pct
    recover_wo = ead_wo * recovery_rate
    collection_cost_wo = recover_wo * collection_cost_pct

    # ── WITH engine (capacity-aware) ──
    reduced_high = baseline_high * (1 - eff_high)
    reduced_medium = baseline_medium * (1 - eff_medium)
    recovery_contacted = min(1.0, recovery_rate + recovery_uplift)

    contacted_defaults = (contacted_high * reduced_high) + (contacted_medium * reduced_medium)
    non_contacted_defaults = (non_contacted_high * baseline_high) + (non_contacted_medium * baseline_medium)
    defaults_w = contacted_defaults + non_contacted_defaults

    contacted_ead = contacted_defaults * avg_exposure
    non_contacted_ead = non_contacted_defaults * avg_exposure
    ead_w = contacted_ead + non_contacted_ead

    credit_loss_w = ead_w * lgd_pct
    recover_w = (contacted_ead * recovery_contacted) + (non_contacted_ead * recovery_rate)
    collection_cost_w = recover_w * collection_cost_pct

    outreach_cost = (contacted_high * cost_per_high) + (contacted_medium * cost_per_medium)

    # ── Net impact ──
    default_reduction_pct = ((defaults_wo - defaults_w) / defaults_wo * 100) if defaults_wo > 0 else 0
    credit_loss_saved = credit_loss_wo - credit_loss_w
    collection_saved = collection_cost_wo - collection_cost_w
    net_savings = (credit_loss_wo + collection_cost_wo) - (credit_loss_w + collection_cost_w) - outreach_cost

    # ── Summary cards ──
    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown(
            f'<div class="impact-group">'
            f'<div class="impact-group-title">Without Engine</div>'
            f'<div class="impact-row"><span class="label">Expected defaults</span>'
            f'<span class="value">{defaults_wo:.1f}</span></div>'
            f'<div class="impact-row"><span class="label">Exposure at Default</span>'
            f'<span class="value">{_fmt_inr(ead_wo)}</span></div>'
            f'<div class="impact-row"><span class="label">Credit loss (EAD x LGD)</span>'
            f'<span class="value">{_fmt_inr(credit_loss_wo)}</span></div>'
            f'<div class="impact-row"><span class="label">Recoveries</span>'
            f'<span class="value">{_fmt_inr(recover_wo)}</span></div>'
            f'<div class="impact-row"><span class="label">Collection cost</span>'
            f'<span class="value">{_fmt_inr(collection_cost_wo)}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with g2:
        st.markdown(
            f'<div class="impact-group">'
            f'<div class="impact-group-title">With Engine (Capacity-Adjusted)</div>'
            f'<div class="impact-row"><span class="label">Expected defaults</span>'
            f'<span class="value">{defaults_w:.1f}</span></div>'
            f'<div class="impact-row"><span class="label">Exposure at Default</span>'
            f'<span class="value">{_fmt_inr(ead_w)}</span></div>'
            f'<div class="impact-row"><span class="label">Credit loss (EAD x LGD)</span>'
            f'<span class="value">{_fmt_inr(credit_loss_w)}</span></div>'
            f'<div class="impact-row"><span class="label">Recoveries</span>'
            f'<span class="value">{_fmt_inr(recover_w)}</span></div>'
            f'<div class="impact-row"><span class="label">Collection cost</span>'
            f'<span class="value">{_fmt_inr(collection_cost_w)}</span></div>'
            f'<div class="impact-row"><span class="label">Outreach cost</span>'
            f'<span class="value">{_fmt_inr(outreach_cost)}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with g3:
        st.markdown(
            f'<div class="impact-group impact-highlight">'
            f'<div class="impact-group-title">Net Impact</div>'
            f'<div class="impact-row"><span class="label">Default reduction</span>'
            f'<span class="impact-saved">{default_reduction_pct:.1f}%</span></div>'
            f'<div class="impact-row"><span class="label">Credit loss avoided</span>'
            f'<span class="impact-saved">{_fmt_inr(credit_loss_saved)}</span></div>'
            f'<div class="impact-row"><span class="label">Collection cost saved</span>'
            f'<span class="impact-saved">{_fmt_inr(collection_saved)}</span></div>'
            f'<div class="impact-row"><span class="label">Outreach cost</span>'
            f'<span class="value">{_fmt_inr(outreach_cost)}</span></div>'
            f'<div class="impact-row"><span class="label"><b>Net savings</b></span>'
            f'<span class="impact-saved" style="font-size:1rem;">{_fmt_inr(net_savings)}</span></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Grouped bar chart ──  
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Credit Loss",
        x=["Without Engine", "With Engine"],
        y=[credit_loss_wo, credit_loss_w],
        marker_color=["#ef4444", "#0d9488"],
        text=[_fmt_inr(credit_loss_wo), _fmt_inr(credit_loss_w)],
        textposition="outside",
        textfont=dict(size=12, family="Inter"),
        offsetgroup=0,
    ))
    fig.add_trace(go.Bar(
        name="Collection Cost",
        x=["Without Engine", "With Engine"],
        y=[collection_cost_wo, collection_cost_w],
        marker_color=["#f59e0b", "#6366f1"],
        text=[_fmt_inr(collection_cost_wo), _fmt_inr(collection_cost_w)],
        textposition="outside",
        textfont=dict(size=12, family="Inter"),
        offsetgroup=1,
    ))
    fig.update_layout(
        title=dict(
            text="Credit Loss and Collection Cost Comparison",
            font=dict(size=14, family="Inter", color="#334155"),
            x=0, xanchor="left",
        ),
        barmode="group",
        yaxis=dict(
            title="Amount (\u20B9)",
            gridcolor="#f1f5f9",
            tickformat=",",
            tickfont=dict(size=11),
        ),
        xaxis=dict(tickfont=dict(size=12, family="Inter")),
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=11)),
        annotations=[dict(
            text=f"Net savings: {_fmt_inr(net_savings)}",
            xref="paper", yref="paper", x=0.98, y=0.95,
            showarrow=False,
            font=dict(size=12, color="#0f766e", family="Inter"),
            bgcolor="#ecfdf3", bordercolor="#bbf7d0", borderwidth=1, borderpad=6,
        )],
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Explanation ──
    st.markdown("")

    # ── Prevention Funnel ──
    st.markdown("##### Prevention Funnel")
    st.markdown(
        '<p class="section-desc">'
        'Operational flow showing how early detection converts high-risk accounts into prevention outcomes.'
        '</p>',
        unsafe_allow_html=True,
    )

    # Compute funnel stages
    flagged_count = eligible_cases
    contacted_count = min(eligible_cases, outreach_capacity)
    accepting_rate = 0.60
    accepting_count = int(contacted_count * accepting_rate)
    defaults_avoided = int(
        (contacted_high * eff_high * accepting_rate) +
        (contacted_medium * eff_medium * accepting_rate)
    )

    funnel_data = {
        "Stage": [
            "Customers Flagged\n(High + Medium Risk)",
            "Customers Contacted\n(This Week)",
            "Accepting Intervention\n(Simulated 60%)",
            "Defaults Avoided\n(Intervention Effective)",
        ],
        "Count": [flagged_count, contacted_count, accepting_count, defaults_avoided],
    }
    funnel_df = pd.DataFrame(funnel_data)

    fig = go.Figure()
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#10b981"]
    for idx, row in funnel_df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["Count"]],
            y=[row["Stage"]],
            orientation="h",
            marker_color=colors[idx],
            text=f"{int(row['Count'])} customers",
            textposition="auto",
            textfont=dict(size=11, color="#ffffff", family="Inter"),
            hovertemplate=f"<b>{row['Stage']}</b><br>Count: {int(row['Count'])}<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        title=dict(
            text="Operational Prevention Funnel",
            font=dict(size=14, family="Inter", color="#334155"),
            x=0, xanchor="left",
        ),
        xaxis=dict(
            title="Number of Customers",
            gridcolor="#f1f5f9",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            tickfont=dict(size=11),
            autorange="reversed",
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        height=340,
        font=dict(family="Inter, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="card" style="margin-top:8px; background:#f9fafb; border:1px solid #e5e7eb;">'
        '<div style="font-size:0.85rem; color:#64748b; line-height:1.7;">'
        '<b>How the Funnel Works:</b> Total flagged accounts are segmented by outreach capacity. '
        'A simulated 60% acceptance rate reflects realistic customer response. '
        'Intervention effectiveness (based on tier) yields the projected default-avoidance count.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ── Explanation ──
    st.markdown(
        '<div class="card" style="margin-top:8px;">'
        '<div style="font-size:0.86rem; color:#475569; line-height:1.7;">'  
        'Early detection reduces delinquency progression by intervening <b>before</b> EMI failure, '
        'lowering credit losses and collection expenses. The engine identifies at-risk customers '
        'weeks ahead of default, enabling proactive outreach — EMI date shifts, restructuring offers, '
        'or budgeting support — that materially reduces the probability of missed payments.'
        '</div>'
        '<div style="font-size:0.82rem; color:#64748b; line-height:1.6; margin-top:12px; '
        'padding-top:10px; border-top:1px solid #f1f5f9;">'
        '<b>Why these projections are realistic:</b>'
        '<ul style="margin:6px 0 0 0; padding-left:20px;">'
        '<li>Loss estimated using LGD (Loss Given Default) methodology, not raw exposure.</li>'
        '<li>Collection cost applied to recovered amounts, not total loss.</li>'
        '<li>Outreach costs included per contacted case, with capacity constraints.</li>'
        '<li>Non-contacted cases beyond weekly capacity remain at baseline rates.</li>'
        '</ul>'
        '</div></div>',
        unsafe_allow_html=True,
    )


# ───────────────────────────────────────────────────────────────────────────────
# How It Works tab
# ───────────────────────────────────────────────────────────────────────────────
def render_how_it_works_tab():
    st.markdown(
        '<p class="section-desc">'
        'This section explains how the Pre-Delinquency Intervention Engine works &mdash; '
        'from data ingestion to intervention delivery. Designed for bank risk teams who need '
        'transparent, explainable early-warning systems.'
        '</p>',
        unsafe_allow_html=True,
    )

    # ── Pipeline steps ──
    cols = st.columns(4)
    steps = [
        ("►", "Ingest Signals",
         "Every week, 7 behavioural features are captured per customer: salary delays, savings changes, "
         "spending patterns, utility payments, lending-app activity, ATM spikes, and autodebit status."),
        ("►", "Compute Risk Score",
         "Each signal is normalised to 0\u20131 and multiplied by an expert-defined weight. "
         "The weighted sum produces a composite risk score (0\u20131) reflecting overall financial stress."),
        ("►", "Classify &amp; Explain",
         "Customers are placed into Low (&lt;0.4), Medium (0.4\u20130.7) or High (&gt;0.7) tiers. "
         "The top contributing signals are surfaced so the RM knows <i>why</i> the customer was flagged."),
        ("►", "Intervene",
         "A rule-based engine maps the risk profile to a recommended action, pre-written customer message, "
         "and preferred outreach channel \u2014 enabling proactive contact before delinquency."),
    ]
    for col, (symbol, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="step-card">'  
                f'<div class="step-num" style="font-size:1.2rem;">{symbol}</div>'  
                f'<h4>{title}</h4>'
                f'<p>{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Signal weights table ──
    st.markdown("##### ⚖ Signal Weights")
    st.markdown(
        '<p class="section-desc">'
        'Each signal carries an expert-assigned weight reflecting its predictive importance for pre-delinquency. '
        'These weights are fixed and transparent \u2014 no black-box models.'
        '</p>',
        unsafe_allow_html=True,
    )

    weight_data = []
    for f in FEATURES:
        weight_data.append({
            "Signal": FEATURE_LABELS[f],
            "Weight": WEIGHTS[f],
        })
    weight_df = pd.DataFrame(weight_data)

    wc1, wc2 = st.columns([1.5, 1])
    with wc1:
        st.dataframe(
            weight_df[["Signal", "Weight"]].style.bar(
                subset=["Weight"], color="#ccfbf1", vmin=0, vmax=0.20
            ).format({"Weight": "{:.0%}"}),
            use_container_width=True,
            hide_index=True,
        )
    with wc2:
        fig = go.Figure(go.Bar(
            y=[FEATURE_LABELS[f] for f in FEATURES],
            x=[WEIGHTS[f] for f in FEATURES],
            orientation="h",
            marker_color="#0d9488",
            text=[f"{WEIGHTS[f]*100:.0f}%" for f in FEATURES],
            textposition="outside",
            textfont=dict(size=11, family="Inter"),
        ))
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=40, t=10, b=10),
            plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, showticklabels=False, range=[0, 0.25]),
            yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("")

    # ── Key design decisions ──
    st.markdown("##### ◆ Key Design Decisions")
    d1, d2, d3 = st.columns(3)
    decisions = [
        ("Explainability First",
         "Every risk score comes with its top contributing signals and plain-English explanations. "
         "No opaque predictions \u2014 RMs can trust and act on the output."),
        ("Weekly Cadence",
         "Signals are refreshed every week to catch emerging stress early. "
         "3-week slope analysis detects whether a customer's trajectory is worsening."),
        ("Empathetic Interventions",
         "Suggested messages are designed to be supportive, not punitive. "
         "The goal is to help customers stay on track, not penalise them."),
    ]
    for col, (title, desc) in zip([d1, d2, d3], decisions):
        with col:
            st.markdown(
                f'<div class="card" style="height:100%;">'
                f'<div style="font-weight:700; font-size:0.92rem; color:#0f172a; margin-bottom:8px;">{title}</div>'
                f'<div style="font-size:0.84rem; color:#64748b; line-height:1.6;">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Compliance guardrails ──
    st.markdown("##### ✓ Compliance Guardrails")
    st.markdown(
        '<div class="card">'
        '<div style="font-size:0.86rem; color:#475569; line-height:1.7;">'
        '<ul style="margin:0; padding-left:20px;">'
        '<li>Uses behavioural deviation signals only &mdash; no protected attributes '
        '(age, gender, caste, religion, geography) are used in scoring.</li>'
        '<li>Explainable top drivers shown for every flag, enabling audit and review '
        'by compliance teams before outreach.</li>'
        '<li>Consistent, rule-based logic applied uniformly across all customers &mdash; '
        'no discretionary overrides in the scoring pipeline.</li>'
        '<li>All outputs (scores, tiers, interventions) are auditable and reproducible '
        'from the same input data.</li>'
        '<li>Designed for responsible banking interventions: supportive outreach, not punitive action.</li>'
        '</ul>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        '<div class="footer">'
        'Built for Hackathon 2025 &bull; Pre-Delinquency Intervention Engine &bull; '
        'Powered by Streamlit + Plotly'
        '</div>',
        unsafe_allow_html=True,
    )


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────
apply_theme()
df = generate_data()
latest = latest_snapshot(df)

render_header()
render_kpis(latest)
render_legend()

weeks = sorted(df["week"].unique().tolist())
customers = sorted(df["customer_id"].unique().tolist())
default_customer = latest.iloc[0]["customer_id"] if not latest.empty else customers[0]
default_customer_idx = customers.index(default_customer) if default_customer in customers else 0

with st.sidebar:
    st.markdown(
        '<div style="text-align:center; padding:24px 0 32px; background:linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius:14px; margin-bottom:12px;">'
        '<div style="font-size:3.2rem; margin-bottom:8px;">🏦</div>'
        '<div style="font-size:1.8rem; font-weight:800; color:#ffffff; letter-spacing:-0.02em;">'
        'PDIE</div>'
        '<div style="font-size:0.72rem; color:#cbd5e1; text-transform:uppercase; letter-spacing:0.08em; margin-top:6px;">'
        'Pre-Delinquency Engine</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    selected_customer = st.selectbox("Customer", customers, index=default_customer_idx)
    selected_week = st.selectbox("Week", weeks, index=len(weeks) - 1)
    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.78rem; color:#64748b; line-height:1.6;">'
        '<b>Quick guide:</b><br>'
        '1. Browse <b>Portfolio</b> for an overview<br>'
        '2. Pick a customer above<br>'
        '3. Switch to <b>Customer View</b> for deep-dive<br>'
        '4. Run <b>Impact Simulation</b> scenarios<br>'
        '5. See <b>How It Works</b> for methodology'
        '</div>',
        unsafe_allow_html=True,
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "Portfolio Overview",
    "Customer View",
    "Impact Simulation",
    "How It Works",
])

with tab1:
    render_portfolio_tab(df, latest)

with tab2:
    render_customer_tab(df, selected_customer, selected_week)

with tab3:
    render_impact_tab(latest)

with tab4:
    render_how_it_works_tab()

render_footer()
