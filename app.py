
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "openthermoai_model.joblib"
METRICS_PATH = BASE / "reports" / "model_validation_metrics.csv"
SCENARIOS_PATH = BASE / "data" / "optimization_scenarios.csv"

st.set_page_config(
    page_title="OpenThermoAI",
    page_icon="🌡️",
    layout="wide"
)

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

bundle = load_bundle()
model = bundle["model"]
FEATURES = bundle["features"]
TARGETS = bundle["targets"]

st.title("🌡️ OpenThermoAI")
st.subheader("AI-powered heat exchanger prediction and optimization")
st.caption(
    "Prototype using synthetic, physically motivated data. "
    "Not yet validated against experimental or CFD results."
)

with st.sidebar:
    st.header("Operating conditions")

    Th = st.slider("Hot inlet temperature (°C)", 65.0, 110.0, 90.0)
    Tc = st.slider("Cold inlet temperature (°C)", 15.0, 35.0, 25.0)
    mh = st.slider("Hot mass flow (kg/s)", 0.15, 2.0, 0.8)
    mc = st.slider("Cold mass flow (kg/s)", 0.15, 2.0, 0.9)
    area = st.slider("Heat transfer area (m²)", 2.0, 20.0, 10.0)
    U = st.slider("Overall heat transfer coefficient (W/m²K)",
                  250.0, 1200.0, 600.0)
    Dh = st.slider("Hot hydraulic diameter (m)", 0.008, 0.025, 0.015)
    Dc = st.slider("Cold hydraulic diameter (m)", 0.008, 0.025, 0.015)
    Lh = st.slider("Hot flow length (m)", 2.0, 12.0, 6.0)
    Lc = st.slider("Cold flow length (m)", 2.0, 12.0, 6.0)

inputs = pd.DataFrame([{
    "Th_in_C": Th,
    "Tc_in_C": Tc,
    "mdot_h_kg_s": mh,
    "mdot_c_kg_s": mc,
    "area_m2": area,
    "U_W_m2K": U,
    "Dh_h_m": Dh,
    "Dh_c_m": Dc,
    "L_h_m": Lh,
    "L_c_m": Lc
}])[FEATURES]

tab1, tab2, tab3 = st.tabs([
    "Prediction", "Optimization", "Validation & data"
])

with tab1:
    st.header("Predicted thermal and hydraulic performance")

    values = model.predict(inputs)[0]
    pred = dict(zip(TARGETS, values))

    c1, c2, c3 = st.columns(3)
    c1.metric("Heat transfer", f"{pred['Q_W']/1000:.2f} kW")
    c2.metric("Hot outlet", f"{pred['Th_out_C']:.2f} °C")
    c3.metric("Cold outlet", f"{pred['Tc_out_C']:.2f} °C")

    c4, c5, c6 = st.columns(3)
    c4.metric("Hot pressure drop", f"{pred['dp_h_Pa']:.0f} Pa")
    c5.metric("Cold pressure drop", f"{pred['dp_c_Pa']:.0f} Pa")
    c6.metric("Hot-side Nusselt number", f"{pred['Nu_h']:.2f}")

    st.write("All predicted outputs")
    st.dataframe(
        pd.DataFrame({
            "Output": TARGETS,
            "Predicted value": values
        }),
        use_container_width=True
    )

    st.download_button(
        "Download prediction as CSV",
        pd.DataFrame([dict(inputs.iloc[0]), pred]).to_csv(index=False),
        file_name="openthermoai_prediction.csv",
        mime="text/csv"
    )

with tab2:
    st.header("Feasible operating-condition search")

    max_dp_h = st.number_input(
        "Maximum hot-side pressure drop (Pa)",
        min_value=1000, max_value=200000, value=60000
    )
    max_dp_c = st.number_input(
        "Maximum cold-side pressure drop (Pa)",
        min_value=1000, max_value=200000, value=60000
    )
    max_tc = st.number_input(
        "Maximum cold outlet temperature (°C)",
        min_value=30.0, max_value=100.0, value=60.0
    )
    n_candidates = st.select_slider(
        "Number of candidate conditions",
        options=[500, 1000, 2500, 5000, 10000],
        value=2500
    )

    if st.button("Run optimization"):
        rng = np.random.default_rng(42)
        candidates = pd.DataFrame({
            "Th_in_C": rng.uniform(65, 110, n_candidates),
            "Tc_in_C": rng.uniform(15, 35, n_candidates),
            "mdot_h_kg_s": rng.uniform(0.15, 2.0, n_candidates),
            "mdot_c_kg_s": rng.uniform(0.15, 2.0, n_candidates),
            "area_m2": rng.uniform(2.0, 20.0, n_candidates),
            "U_W_m2K": rng.uniform(250, 1200, n_candidates),
            "Dh_h_m": rng.uniform(0.008, 0.025, n_candidates),
            "Dh_c_m": rng.uniform(0.008, 0.025, n_candidates),
            "L_h_m": rng.uniform(2.0, 12.0, n_candidates),
            "L_c_m": rng.uniform(2.0, 12.0, n_candidates)
        })

        outputs = pd.DataFrame(
            model.predict(candidates[FEATURES]),
            columns=TARGETS
        )
        results = pd.concat([candidates, outputs], axis=1)

        feasible = results[
            (results["dp_h_Pa"] <= max_dp_h) &
            (results["dp_c_Pa"] <= max_dp_c) &
            (results["Tc_out_C"] <= max_tc) &
            (results["Tc_out_C"] <= results["Th_in_C"]) &
            (results["Th_out_C"] >= results["Tc_in_C"])
        ].sort_values("Q_W", ascending=False)

        if feasible.empty:
            st.warning(
                "No feasible candidates found. Relax the constraints "
                "or increase the candidate count."
            )
        else:
            st.success(
                f"Found {len(feasible)} feasible candidates."
            )
            st.dataframe(feasible.head(20), use_container_width=True)
            st.metric(
                "Best predicted heat transfer",
                f"{feasible.iloc[0]['Q_W']/1000:.2f} kW"
            )
            st.download_button(
                "Download optimization results",
                feasible.to_csv(index=False),
                file_name="openthermoai_optimization.csv",
                mime="text/csv"
            )

with tab3:
    st.header("Model validation and data")

    st.write("Selected model:", bundle["selected_model"])
    st.write("Training data:", bundle["training_data_type"])

    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics, use_container_width=True)
        st.download_button(
            "Download validation metrics",
            metrics.to_csv(index=False),
            file_name="model_validation_metrics.csv",
            mime="text/csv"
        )
    else:
        st.info("Validation metrics file not found.")

    st.write("### Model assumptions")
    for item in bundle["assumptions"]:
        st.write("- " + item)

    st.warning(
        "This app uses a surrogate trained on synthetic data. "
        "Do not use its outputs as certified equipment-design values. "
        "Validate against experiments, manufacturer data, or CFD "
        "before engineering deployment."
    )
