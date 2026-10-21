# OpenThermoAI
# OpenThermoAI: AI-Powered Heat Exchanger Prediction and Optimization

OpenThermoAI is an open-source engineering project that combines machine learning, thermodynamics, heat transfer, and optimization to predict and improve heat exchanger performance. The initial prototype uses synthetic data generated from simplified engineering equations to predict heat-transfer rate, outlet temperatures, pressure drops, Nusselt numbers, and heat-transfer coefficients. It compares Random Forest and Extra Trees regression models, evaluates predictive performance using MAE, RMSE, and R², and identifies promising operating conditions subject to engineering constraints. A Streamlit web application provides interactive predictions, optimization results, and model validation metrics. Developed in Python and Google Colab, OpenThermoAI is designed to evolve toward validation using computational fluid dynamics (CFD) simulations from ANSYS Fluent or OpenFOAM and, where available, experimental measurements. The long-term objective is to support faster thermal-system analysis, energy efficiency improvements, and data-driven engineering decisions. **Current limitation:** predictions are trained on synthetic data and have not yet been validated against experimental measurements or independent CFD results. Therefore, the prototype is intended for research, learning, and demonstration rather than certified equipment design.

## Key Features

- Synthetic heat exchanger dataset generation.
- Machine learning model training and comparison.
- Prediction of thermal and hydraulic performance.
- Constrained operating-condition optimization.
- Model evaluation using MAE, RMSE, and R².
- Interactive Streamlit dashboard.
- CSV export of predictions and optimization scenarios.

## Technologies

Python, Google Colab, NumPy, Pandas, Scikit-learn, Matplotlib, Joblib, and Streamlit.

## Project Structure

```text
OpenThermoAI/
├── app.py
├── OpenThermoAI_Heat_Exchanger.ipynb
├── requirements.txt
├── README.md
├── data/
│   ├── synthetic_heat_exchanger_data.csv
│   └── optimization_scenarios.csv
├── models/
│   └── openthermoai_model.joblib
└── reports/
    └── model_validation_metrics.csv
```

## Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```

## Future Development

Planned improvements include real CFD dataset integration, external model validation, uncertainty quantification, improved heat exchanger correlations, and multi-objective optimization of heat transfer, pressure losses, and energy consumption.

## Disclaimer

OpenThermoAI is an experimental engineering prototype. Synthetic-data performance metrics do not establish accuracy for real-world equipment. Validate all predictions against suitable CFD simulations, experimental measurements, and applicable engineering standards before practical use.
