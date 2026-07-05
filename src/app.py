import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

# ---------------------------------------------------------------------------
# Load model + feature order (trained in notebooks/churn_updated.ipynb, Section 10)
# Resolved relative to this file's own location (src/../models/) rather than
# whatever directory the app happens to be launched from - this matters because
# `streamlit run src/app.py` can be run from different working directories
# depending on how it's deployed.
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

@st.cache_resource
def load_model():
    model = joblib.load(MODELS_DIR / "churn_model.pkl")
    feature_order = joblib.load(MODELS_DIR / "feature_order.pkl")
    return model, feature_order

model, FEATURE_ORDER = load_model()


def build_features(row):
    """Takes a dict of raw customer inputs and builds the exact feature set
    the model was trained on - same logic as the notebook's feature engineering."""
    geography_germany = 1 if row['Geography'] == 'Germany' else 0
    geography_spain = 1 if row['Geography'] == 'Spain' else 0
    gender = 1 if row['Gender'] == 'Male' else 0

    at_risk = int(row['NumOfProducts'] == 1 and row['IsActiveMember'] == 0)
    age_activity_risk = int(40 <= row['Age'] < 60 and row['IsActiveMember'] == 0)

    features = {
        'Gender': gender,
        'Age': row['Age'],
        'Balance': row['Balance'],
        'NumOfProducts': row['NumOfProducts'],
        'IsActiveMember': row['IsActiveMember'],
        'Geography_Germany': geography_germany,
        'Geography_Spain': geography_spain,
        'AtRiskCustomer': at_risk,
        'AgeActivityRisk': age_activity_risk,
    }
    return pd.DataFrame([features])[FEATURE_ORDER]


def explain_flags(row, probability):
    """Plain-language reasons, based on the same signals the model relies on
    most (Age, Balance, IsActiveMember - see the notebook's feature importance
    section) plus the two engineered risk flags actually in the model."""
    reasons = []
    if row['IsActiveMember'] == 0 and row['Balance'] > 0:
        reasons.append("Inactive member with a balance on the account — the strongest churn signal in the data.")
    if row['NumOfProducts'] == 1 and row['IsActiveMember'] == 0:
        reasons.append("Only 1 product and inactive — flagged as an at-risk customer.")
    if 40 <= row['Age'] < 60 and row['IsActiveMember'] == 0:
        reasons.append("Age 40–60 and inactive — this combination showed higher churn in the training data.")
    if row['Geography'] == 'Germany':
        reasons.append("Germany had the highest churn rate by country in the training data.")
    if not reasons:
        reasons.append("No strong risk flags — this customer looks similar to typical retained customers.")
    return reasons


tab1, tab2, tab3 = st.tabs(["🔍 Single Customer", "📋 Batch Upload", "💰 Business Impact"])

# ---------------------------------------------------------------------------
# TAB 1 - Single customer prediction
# ---------------------------------------------------------------------------
with tab1:
    st.header("Predict churn for one customer")
    st.caption("Enter a customer's details to get a churn probability, using the final "
               "XGBoost model from the notebook (test F1: 0.58, recall: 49%).")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 92, 40)
        balance = st.number_input("Balance ($)", min_value=0.0, max_value=300000.0, value=76000.0, step=1000.0)
        num_products = st.selectbox("Number of products", [1, 2, 3, 4], index=0)
    with col2:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        is_active = st.selectbox("Active member?", ["Yes", "No"]) == "Yes"

    row = {
        'Age': age,
        'Balance': balance,
        'NumOfProducts': num_products,
        'Geography': geography,
        'Gender': gender,
        'IsActiveMember': 1 if is_active else 0,
    }

    # Recomputes on every input change - no button needed, so the result never
    # goes stale relative to what's on screen.
    X = build_features(row)
    proba = model.predict_proba(X)[0, 1]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Churn probability", f"{proba*100:.1f}%")
        if proba >= 0.5:
            st.error("Predicted: Will churn")
        else:
            st.success("Predicted: Will stay")
        st.caption("Threshold: 50% (the notebook's default; see the Business Impact "
                   "tab for why a different threshold might make more sense in practice).")

    with c2:
        st.subheader("Why")
        for reason in explain_flags(row, proba):
            st.write(f"- {reason}")

    if is_active:
        st.info("This customer is an active member — in the final model, activity status "
                 "is the single strongest signal, so probability will stay low here regardless "
                 "of other inputs. Toggle 'Active member?' to No to see the other features "
                 "start to matter.")

# ---------------------------------------------------------------------------
# TAB 2 - Batch upload
# ---------------------------------------------------------------------------
with tab2:
    st.header("Score a list of customers")
    st.caption("Upload a CSV with columns: Age, Balance, NumOfProducts, Geography, Gender, IsActiveMember "
               "(IsActiveMember as 0/1). Get every customer ranked by churn probability, highest risk first — "
               "this is the list a retention team would actually work from.")

    sample = pd.DataFrame({
        'Age': [45, 29], 'Balance': [95000, 0], 'NumOfProducts': [1, 2],
        'Geography': ['Germany', 'France'], 'Gender': ['Female', 'Male'], 'IsActiveMember': [0, 1]
    })
    st.download_button("Download a sample CSV", sample.to_csv(index=False), "sample_customers.csv")

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        required_cols = ['Age', 'Balance', 'NumOfProducts', 'Geography', 'Gender', 'IsActiveMember']
        missing = [c for c in required_cols if c not in batch_df.columns]

        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            feature_rows = []
            for _, r in batch_df.iterrows():
                feature_rows.append(build_features(r).iloc[0])
            X_batch = pd.DataFrame(feature_rows)[FEATURE_ORDER]

            batch_df['Churn Probability'] = model.predict_proba(X_batch)[:, 1]
            batch_df = batch_df.sort_values('Churn Probability', ascending=False).reset_index(drop=True)
            batch_df['Churn Probability'] = (batch_df['Churn Probability'] * 100).round(1)

            st.success(f"Scored {len(batch_df)} customers.")
            st.dataframe(batch_df, use_container_width=True)
            st.download_button("Download results", batch_df.to_csv(index=False), "scored_customers.csv")

            # stash for the business impact tab
            st.session_state['batch_df'] = batch_df

# ---------------------------------------------------------------------------
# TAB 3 - Business impact (same logic as notebook Section 7)
# ---------------------------------------------------------------------------
with tab3:
    st.header("Is it worth acting on these predictions?")
    st.caption("Same sensitivity-analysis logic as the notebook's business impact section: "
               "real retention-cost and customer-value figures weren't available, so this "
               "uses adjustable assumptions instead of one fixed number.")

    c1, c2 = st.columns(2)
    with c1:
        retention_cost = st.slider("Retention offer cost per customer ($)", 20, 200, 75, step=5)
    with c2:
        customer_value = st.slider("Value of keeping a customer ($)", 100, 1500, 500, step=50)

    if 'batch_df' in st.session_state:
        batch_df = st.session_state['batch_df']
        st.subheader("Apply to your uploaded batch")
        top_n = st.slider("If you can only contact the top N highest-risk customers",
                           1, len(batch_df), min(50, len(batch_df)))

        top_customers = batch_df.head(top_n)
        # assume contacting these saves anyone genuinely at high risk (>=50% predicted probability)
        likely_saved = (top_customers['Churn Probability'] >= 50).sum()
        cost = top_n * retention_cost
        value_saved = likely_saved * customer_value
        net = value_saved - cost

        m1, m2, m3 = st.columns(3)
        m1.metric("Customers contacted", top_n)
        m2.metric("Estimated cost", f"${cost:,.0f}")
        m3.metric("Estimated net value", f"${net:,.0f}", delta=f"{likely_saved} likely saved")

        st.caption("Assumes a retention offer saves a customer with certainty if they were genuinely "
                   "going to churn (a simplification — see the notebook for the caveats on this "
                   "assumption, including why 'contact everyone' isn't automatically the best strategy).")
    else:
        st.info("Upload a batch of customers in the 'Batch Upload' tab to see the estimated "
                "value of acting on this model's predictions for your specific list.")