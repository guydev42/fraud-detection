"""Streamlit dashboard for credit card fraud detection."""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, average_precision_score, precision_score,
    recall_score, f1_score,
)

PROJECT_DIR = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_DIR)

from src.data_loader import generate_fraud_data, load_and_prepare
from src.model import _get_models, RANDOM_STATE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Fraud detection", layout="wide")

GOLD = "#E8C230"
NAVY = "#3B6FD4"


@st.cache_data
def load_data():
    path = os.path.join(PROJECT_DIR, "data", "fraud_transactions.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return generate_fraud_data()


@st.cache_resource
def train_models(df):
    """Train all models and return results."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE

    feature_cols = [c for c in df.columns if c != "is_fraud"]
    X = df[feature_cols].values.astype(float)
    y = df["is_fraud"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    models_config = _get_models()
    results = {}
    trained = {}

    for name, config in models_config.items():
        model = config["model"]
        Xtr = X_train_scaled if config["needs_scaling"] else X_train_res
        Xte = X_test_scaled if config["needs_scaling"] else X_test

        model.fit(Xtr, y_train_res)
        y_prob = model.predict_proba(Xte)[:, 1]
        y_pred = model.predict(Xte)

        results[name] = {
            "y_prob": y_prob,
            "auc_roc": roc_auc_score(y_test, y_prob),
            "pr_auc": average_precision_score(y_test, y_prob),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }
        trained[name] = {
            "model": model,
            "needs_scaling": config["needs_scaling"],
        }

    return results, trained, X_test, X_test_scaled, y_test, feature_cols, scaler


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Transaction scoring", "Model comparison", "Threshold tuning", "SHAP analysis"],
)

df = load_data()
results, trained, X_test, X_test_scaled, y_test, feature_cols, scaler = train_models(df)

best_name = max(results, key=lambda n: results[n]["auc_roc"])


# ---------------------------------------------------------------------------
# Page: Transaction scoring
# ---------------------------------------------------------------------------
if page == "Transaction scoring":
    st.title("Transaction fraud scoring")
    st.markdown("Adjust transaction features to see real-time fraud probability.")

    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.slider("Transaction amount ($)", 0.50, 5000.0, 150.0, step=5.0)
        time_hour = st.slider("Hour of day", 0, 23, 14)
        merchant_cat = st.selectbox("Merchant category", list(range(10)))
        is_weekend = st.checkbox("Weekend transaction")
    with col2:
        dist_home = st.slider("Distance from home (mi)", 0.0, 200.0, 10.0, step=1.0)
        dist_last = st.slider("Distance from last transaction (mi)", 0.0, 200.0, 5.0, step=1.0)
        ratio_median = st.slider("Ratio to median purchase", 0.1, 20.0, 1.0, step=0.1)
    with col3:
        is_night = st.checkbox("Night transaction (22:00-06:00)")
        txn_last_hour = st.slider("Transactions in last hour", 0, 15, 1)
        txn_last_day = st.slider("Transactions in last day", 0, 40, 5)

    features = np.array([[
        amount, time_hour, merchant_cat, dist_home, dist_last,
        ratio_median, int(is_weekend), int(is_night),
        txn_last_hour, txn_last_day,
    ]])

    info = trained[best_name]
    X_input = scaler.transform(features) if info["needs_scaling"] else features
    prob = info["model"].predict_proba(X_input)[0, 1]

    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Fraud probability", f"{prob:.1%}")
    col_b.metric("Model", best_name)
    col_c.metric("Decision", "BLOCK" if prob > 0.5 else "APPROVE",
                 delta="High risk" if prob > 0.5 else "Low risk",
                 delta_color="inverse")

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": "Fraud risk score"},
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": NAVY},
            "steps": [
                {"range": [0, 30], "color": "#22c55e"},
                {"range": [30, 70], "color": GOLD},
                {"range": [70, 100], "color": "#ef4444"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Model comparison
# ---------------------------------------------------------------------------
elif page == "Model comparison":
    st.title("Model comparison")

    # Metrics table
    metrics_df = pd.DataFrame({
        name: {k: v for k, v in r.items() if k != "y_prob"}
        for name, r in results.items()
    }).T.round(4)
    st.dataframe(metrics_df, use_container_width=True)

    # ROC curves
    st.subheader("ROC curves")
    fig = go.Figure()
    for name, r in results.items():
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name} (AUC={r['auc_roc']:.3f})",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"), name="Random",
    ))
    fig.update_layout(
        xaxis_title="False positive rate", yaxis_title="True positive rate",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    # PR curves
    st.subheader("Precision-recall curves")
    fig = go.Figure()
    for name, r in results.items():
        prec, rec, _ = precision_recall_curve(y_test, r["y_prob"])
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=f"{name} (PR-AUC={r['pr_auc']:.3f})",
        ))
    fig.update_layout(
        xaxis_title="Recall", yaxis_title="Precision",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Threshold tuning
# ---------------------------------------------------------------------------
elif page == "Threshold tuning":
    st.title("Threshold tuning with cost analysis")

    st.markdown("Adjust the costs to find the optimal decision threshold.")

    col1, col2 = st.columns(2)
    with col1:
        fn_cost = st.number_input("Cost per missed fraud (FN)", 50, 5000, 500, step=50)
    with col2:
        fp_cost = st.number_input("Cost per false alarm (FP)", 5, 500, 25, step=5)

    y_prob = results[best_name]["y_prob"]
    thresholds = np.arange(0.05, 0.96, 0.01)
    records = []

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, y_pred_t)
        tn, fp, fn, tp = cm.ravel()
        total_cost = (fn * fn_cost) + (fp * fp_cost)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        records.append({
            "Threshold": round(t, 3),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Recall": round(rec, 4),
            "Precision": round(prec, 4),
            "FPR": round(fpr, 4),
            "Total cost ($)": total_cost,
        })

    cost_df = pd.DataFrame(records)
    optimal_idx = cost_df["Total cost ($)"].idxmin()
    optimal = cost_df.loc[optimal_idx]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Optimal threshold", f"{optimal['Threshold']:.3f}")
    col_b.metric("Recall", f"{optimal['Recall']:.1%}")
    col_c.metric("False positive rate", f"{optimal['FPR']:.1%}")
    col_d.metric("Total cost", f"${int(optimal['Total cost ($)']):,}")

    # Cost curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cost_df["Threshold"], y=cost_df["Total cost ($)"],
        mode="lines", name="Total cost", line=dict(color=NAVY, width=2),
    ))
    fig.add_vline(x=optimal["Threshold"], line_dash="dash", line_color="red",
                  annotation_text=f"Optimal: {optimal['Threshold']:.3f}")
    fig.update_layout(
        xaxis_title="Threshold", yaxis_title="Total cost ($)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Recall vs FPR curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cost_df["Threshold"], y=cost_df["Recall"],
        mode="lines", name="Recall", line=dict(color="#22c55e", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=cost_df["Threshold"], y=cost_df["FPR"],
        mode="lines", name="False positive rate", line=dict(color="#ef4444", width=2),
    ))
    fig.add_vline(x=optimal["Threshold"], line_dash="dash", line_color="red")
    fig.update_layout(
        xaxis_title="Threshold", yaxis_title="Rate",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(cost_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: SHAP analysis
# ---------------------------------------------------------------------------
elif page == "SHAP analysis":
    st.title("SHAP explainability")

    import shap

    info = trained[best_name]
    model = info["model"]
    X = X_test_scaled if info["needs_scaling"] else X_test

    sample_size = min(300, X.shape[0])
    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(X.shape[0], sample_size, replace=False)
    X_sample = X[idx]

    if info["needs_scaling"]:
        explainer = shap.LinearExplainer(model, X_sample)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Global feature importance
    st.subheader("Global feature importance")
    mean_abs = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Mean |SHAP|": mean_abs,
    }).sort_values("Mean |SHAP|", ascending=True)

    fig = px.bar(importance_df, x="Mean |SHAP|", y="Feature", orientation="h",
                 color_discrete_sequence=[NAVY])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Individual prediction
    st.subheader("Individual transaction explanation")
    probs = model.predict_proba(X_sample)[:, 1]

    tx_index = st.slider("Select transaction index", 0, sample_size - 1, 0)

    st.metric("Fraud probability", f"{probs[tx_index]:.1%}")

    # Feature contributions
    contrib_df = pd.DataFrame({
        "Feature": feature_cols,
        "SHAP value": shap_vals[tx_index],
        "Feature value": X_sample[tx_index],
    }).sort_values("SHAP value", key=abs, ascending=True)

    fig = px.bar(contrib_df.tail(10), x="SHAP value", y="Feature", orientation="h",
                 color="SHAP value",
                 color_continuous_scale=["#3B6FD4", "#cccccc", "#E8C230"])
    fig.update_layout(height=400, title="Top 10 feature contributions")
    st.plotly_chart(fig, use_container_width=True)

    # Feature values table
    st.subheader("Transaction details")
    details = pd.DataFrame({
        "Feature": feature_cols,
        "Value": X_sample[tx_index],
        "SHAP contribution": shap_vals[tx_index],
    })
    st.dataframe(details, use_container_width=True, hide_index=True)
