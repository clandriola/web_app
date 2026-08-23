import streamlit as st
import pandas as pd
from joblib import load

from modeling_aux import (
    feature_eng,
    rule_based_event,
    rule_based_steering,
    rule_based_acceleration,
    clean_X,
    build_avoidance_label,
    plot_conflict
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Near Miss Sequence of Events Extraction",
    page_icon="🚗",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }

    .prediction-box {
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }

    .vehicle-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .event-label {
        font-size: 1rem;
        line-height: 1.6;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Near Miss Sequence of Events Extraction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Trajectory-based sequence-of-events extraction using a hybrid framework of rule-based and learning-based models'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# EVENT MAPPINGS
# ============================================================

e1_map = {
    1: "Going Straight",
    2: "Decelerating in Road",
    3: "Accelerating in Road",
    4: "Passing or Overtaking Another Vehicle",
    5: "Changing Lanes",
    6: "Merging"
}

e2_map = {
    1: "This Vehicle Decelerating",
    2: "Over the Lane Line on Left Side of Travel Lane",
    3: "Over the Lane Line on Right Side of Travel Lane",
    4: "Traveling in Same Direction with Lower Steady Speed",
    5: "Traveling in Same Direction while Decelerating",
    6: "Traveling in Same Direction with Higher Speed",
    7: "From Adjacent Lane (Same Direction)-Over Left Lane Line",
    8: "From Adjacent Lane (Same Direction)-Over Right Lane Line"
}

e3_map = {
    1: "No Avoidance Maneuver",
    2: "Steering Left",
    3: "Steering Right",
    4: "Braking",
    5: "Braking and Steering Left",
    6: "Braking and Steering Right",
    7: "Accelerated",
    8: "Accelerating and Steering Left",
    9: "Accelerating and Steering Right"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_models():

    artifacts = load("hybrid_framework.joblib")

    return artifacts


artifacts = load_models()

rf_e1e2 = artifacts["rf_e1e2"]

rf_steer_v1 = artifacts["rf_steer_v1"]
rf_steer_v2 = artifacts["rf_steer_v2"]

rf_acc_v1 = artifacts["rf_acc_v1"]
rf_acc_v2 = artifacts["rf_acc_v2"]

le_e1e2 = artifacts["le_e1e2"]

le_steer_v1 = artifacts["le_steer_v1"]
le_steer_v2 = artifacts["le_steer_v2"]

le_acc_v1 = artifacts["le_acc_v1"]
le_acc_v2 = artifacts["le_acc_v2"]

FEATURE_COLUMNS = artifacts["feature_columns"]


# ============================================================
# MODEL STATUS
# ============================================================

with st.expander("Model information"):

    st.success("Hybrid models loaded successfully.")

    st.write(
        f"Number of input features: **{len(FEATURE_COLUMNS)}**"
    )


# ============================================================
# LOAD TRAJECTORY DATA
# ============================================================

st.divider()

st.header("1. Trajectory Data")

uploaded_file = st.file_uploader(
    "Upload trajectory pair CSV",
    type=["csv"]
)


# ------------------------------------------------------------
# Temporary option for testing with a local file
# ------------------------------------------------------------

if uploaded_file is not None:

    df_pair = pd.read_csv(uploaded_file)

else:

    pair_id = ("6390f826dccd9b1228bf3d5e_6390f82edccd9b1228bf3d79")
    try:
        df_pair = pd.read_csv(pair_id + ".csv")
        st.info(
            "Using the default trajectory pair for testing."
        )

    except FileNotFoundError:
        st.warning(
            "Upload a trajectory pair CSV to begin."
        )
        st.stop()


# ============================================================
# TRAJECTORY INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Rows",
        len(df_pair)
    )

with col2:
    st.metric(
        "Columns",
        len(df_pair.columns)
    )

with col3:
    st.metric(
        "Vehicles",
        "2"
    )


# ============================================================
# TRAJECTORY PLOT
# ============================================================

st.divider()

st.header("2. Trajectory Visualization")

fig, _ = plot_conflict(
    df_pair,
    "id1_id2",
    3,
    save_path=None
)

st.pyplot(
    fig,
    use_container_width=True
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

df_processed = feature_eng(df_pair)

feature_row = (
    df_processed[FEATURE_COLUMNS]
    .iloc[0]
    .to_dict()
)


# ============================================================
# HYBRID PREDICTION
# ============================================================

def predict_hybrid(feature_row, df_pair):

    sample = pd.DataFrame([feature_row])

    sample = sample.reindex(
        columns=FEATURE_COLUMNS,
        fill_value=0
    )

    sample = clean_X(sample)


    # ========================================================
    # STEP 1 - EVENT
    # ========================================================

    rb_event = rule_based_event(df_pair)

    label_e1e2 = (
        rb_event[:2]
        + rb_event[3:5]
    )

    if label_e1e2 == "NA":

        label_e1e2 = (
            le_e1e2.inverse_transform(
                rf_e1e2.predict(sample)
            )[0]
        )

        rb_event = (
            label_e1e2[:2]
            + ","
            + label_e1e2[2:4]
        )


    # ========================================================
    # STEP 2 - STEERING
    # ========================================================

    steer_v1_rb, steer_v2_rb = (
        rule_based_steering(
            df_pair,
            rb_event
        ).split(",")
    )

    if steer_v1_rb == "NA":

        steer_v1 = (
            le_steer_v1.inverse_transform(
                rf_steer_v1.predict(sample)
            )[0]
        )

    else:

        steer_v1 = steer_v1_rb


    if steer_v2_rb == "NA":

        steer_v2 = (
            le_steer_v2.inverse_transform(
                rf_steer_v2.predict(sample)
            )[0]
        )

    else:

        steer_v2 = steer_v2_rb


    # ========================================================
    # STEP 3 - ACCELERATION
    # ========================================================

    acc_v1_rb, acc_v2_rb = (
        rule_based_acceleration(
            df_pair,
            rb_event
        ).split(",")
    )

    if acc_v1_rb == "NA":

        acc_v1 = (
            le_acc_v1.inverse_transform(
                rf_acc_v1.predict(sample)
            )[0]
        )

    else:

        acc_v1 = acc_v1_rb


    if acc_v2_rb == "NA":

        acc_v2 = (
            le_acc_v2.inverse_transform(
                rf_acc_v2.predict(sample)
            )[0]
        )

    else:

        acc_v2 = acc_v2_rb


    # ========================================================
    # STEP 4 - E3 LABELS
    # ========================================================

    label_e3_v1 = build_avoidance_label(
        steer_v1,
        acc_v1
    )

    label_e3_v2 = build_avoidance_label(
        steer_v2,
        acc_v2
    )


    # ========================================================
    # STEP 5 - FINAL LABEL
    # ========================================================

    final_label = (
        label_e1e2[:2]
        + label_e3_v1
        + ","
        + label_e1e2[2:4]
        + label_e3_v2
    )


    return {
        "event_prediction": label_e1e2,

        "steer_v1": steer_v1,
        "steer_v2": steer_v2,

        "acc_v1": acc_v1,
        "acc_v2": acc_v2,

        "label_e3_v1": label_e3_v1,
        "label_e3_v2": label_e3_v2,

        "final_label": final_label
    }


prediction = predict_hybrid(
    feature_row,
    df_pair
)


# ============================================================
# PREDICTION RESULTS
# ============================================================

st.divider()

st.header("3. Sequence of Events")


# ------------------------------------------------------------
# Decode final label
# ------------------------------------------------------------

final_label = prediction["final_label"]

v1_label, v2_label = final_label.split(",")

v1_e1, v1_e2, v1_e3 = map(int, v1_label)
v2_e1, v2_e2, v2_e3 = map(int, v2_label)


# ============================================================
# VEHICLE RESULTS
# ============================================================

vehicle1, vehicle2 = st.columns(2)


with vehicle1:

    st.markdown(
        '<div class="prediction-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="vehicle-title">Vehicle 1</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="event-label">
        <b>E1:</b> {e1_map[v1_e1]}<br>
        <b>E2:</b> {e2_map[v1_e2]}<br>
        <b>E3:</b> {e3_map[v1_e3]}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


with vehicle2:

    st.markdown(
        '<div class="prediction-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="vehicle-title">Vehicle 2</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="event-label">
        <b>E1:</b> {e1_map[v2_e1]}<br>
        <b>E2:</b> {e2_map[v2_e2]}<br>
        <b>E3:</b> {e3_map[v2_e3]}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# FINAL LABEL
# ============================================================

st.subheader("Final Label")

st.code(
    prediction["final_label"],
    language=None
)


# ============================================================
# MODEL DETAILS
# ============================================================

with st.expander("Prediction details"):

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Event:**", prediction["event_prediction"])

        st.write(
            "**Vehicle 1 Steering:**",
            prediction["steer_v1"]
        )

        st.write(
            "**Vehicle 1 Acceleration:**",
            prediction["acc_v1"]
        )

        st.write(
            "**Vehicle 1 E3:**",
            prediction["label_e3_v1"]
        )


    with col2:

        st.write(
            "**Vehicle 2 Steering:**",
            prediction["steer_v2"]
        )

        st.write(
            "**Vehicle 2 Acceleration:**",
            prediction["acc_v2"]
        )

        st.write(
            "**Vehicle 2 E3:**",
            prediction["label_e3_v2"]
        )


# ============================================================
# RAW DATA
# ============================================================

with st.expander("View trajectory data"):

    st.dataframe(
        df_pair,
        use_container_width=True
    )

