# OpenThermoAI

OpenThermoAI is an open-source prototype for AI-assisted heat exchanger
prediction and operating-condition optimization.

## Features
- Predict heat transfer, outlet temperatures, pressure drop, Nusselt
  numbers, and heat-transfer coefficients.
- Compare Random Forest and Extra Trees on held-out synthetic data.
- Search sampled operating conditions subject to pressure-drop and
  temperature constraints.
- Explore model validation metrics through Streamlit.

## Data and limitations
The initial dataset is synthetic and generated using a simplified
counterflow effectiveness-NTU thermal model and simplified hydraulic
correlations. It is not experimental data or validated CFD output.
The model should not be used for certified engineering design.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
