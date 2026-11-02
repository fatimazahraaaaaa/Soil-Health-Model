# Explainable Long-Term Soil Health Forecasting

Machine-learning pipeline built on the LUCAS 2018 topsoil survey: it builds a Sustainable Health Index (SHI)
for each sample point, predicts it with XGBoost and Random Forest, explains the models with SHAP,
and runs what-if scenario analysis.

## Pipeline
| Stage | Description |
|---|---|
| 1 | Data ingestion, cleaning & merging |
| 2 | Exploratory data analysis |
| 3 | Imputation & log transforms |
| 4 | Sustainable Health Index (SHI) |
| 5 | Feature engineering |
| 6 | XGBoost & Random Forest (regression + classification) |
| 7 | SHAP explainability & diagnostics |
| 8 | What-if scenario analysis |
| 9 | Save outputs |

## Repository structure
```
notebooks/Soil_health_notebook.ipynb   full pipeline, with outputs
src/data_preparation.py                stage 1: load, clean, merge the 3 raw files
src/soil_health_pipeline.py            stages 2-9 as reusable functions
report/ML_Report.pdf                   project report
docs/state_of_art.xlsx                 state-of-the-art review
data/                                  put the raw CSVs here (see data/README.md)
```

## Getting started
```bash
pip install -r requirements.txt

# 1. Build the merged dataset
python src/data_preparation.py \
  --core data/LUCAS-SOIL-2018.csv \
  --bd data/BulkDensity_2018_final-2.csv \
  --texture data/LUCAS_Text_All_10032025.csv \
  --out data/LUCAS_master.csv
```
Then open `notebooks/Soil_health_notebook.ipynb` (originally run on Google Colab; place the raw CSVs and the two
`src/` scripts next to the notebook, or adjust the paths in the first cells).

## Data
LUCAS 2018 topsoil data, European Soil Data Centre (ESDAC). Not redistributed here, see `data/README.md`.

## Author
ZaHra, Master IPS, Faculté des Sciences, Université Mohammed V de Rabat.
