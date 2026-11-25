# Predicting US Accident Severity and Clearing Time

CDS 4003 - Data Science for Business Applications | Higher Colleges of Technology
Instructor: Dr Zakea Ilagure | CRN 16009 | Group 7

**Live app:** https://cds4003-us-accidents.streamlit.app

## Business problem

When an accident is reported, a traffic operator has to decide quickly: send extra crews,
reroute traffic, or just monitor. The right choice depends on two things that are not known yet:
how serious the traffic delay will be (Severity 1 to 4) and how many minutes the road will stay
blocked. We built two models that predict both from information available at the time of the
report: place, time, weather and nearby road features.

Our recommendation: use the models as a **pilot with a human checking each alert**, and not for
automatic rerouting. The models catch about half of the Severity 4 accidents, but most of their
Severity 4 alerts are false alarms.

## Team

| Member | Student ID | Stages |
|---|---|---|
| Rashed Abdulla Al Qersi Al Ali | H00504272 | 1 Business problem, 2 Data selection, 3 Data cleaning. Also created the repository and the Streamlit app. |
| Ahmed Salem ALHebsi | H00478241 | 4 EDA and feature engineering, 5 Model selection |
| Khalifa Ali Alketbi | H00444458 | 6 Tuning and evaluation, 7 Deployment and business value |

See [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md) for details. Ahmed uploaded his EDA notebook (01b_eda_findings.ipynb) from his own account.

## Dataset

- **Name:** US Accidents (March 2023 version), by Sobhan Moosavi et al.
- **Source:** https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents
- **Licence:** CC BY-NC-SA 4.0 (non-commercial and study use, share alike)
- **What we used:** a 500,000-row sample of that dataset (497,097 rows after cleaning)
- **How to get it:** see [data/README.md](data/README.md). The file is 189 MB, so it is not on
  GitHub. A 1,000-row sample is in `data/sample_accidents.csv`.

## Folder structure

```
data/        dataset notes, a small sample (the full CSV goes here, not on GitHub)
notebooks/   01_data_preparation.ipynb, 01b_eda_findings.ipynb, 02_model_building.ipynb
src/         prep.py (data preparation function), train_light_model.py (model for the app)
app/         app.py (Streamlit app)
models/      severity_model.joblib, duration_model.joblib, model_info.json
docs/        reports (CLO2, CLO3) and the final presentation
```

## How to run

```bash
pip install -r requirements.txt

# 1. Put US_Accidents_March23_sampled_500k.csv in the data/ folder (see data/README.md)

# 2. Notebooks (run them from the notebooks/ folder, they use ../data and ../src)
jupyter notebook notebooks/

# 3. Train the light models again (optional, the trained files are already in models/)
python src/train_light_model.py data/US_Accidents_March23_sampled_500k.csv

# 4. Start the app
streamlit run app/app.py
```

The app is deployed on Streamlit Community Cloud from this repository (main file: `app/app.py`).

## Results summary

Test set of 99,420 accidents that the models never saw (20% split, seed 42).

| | Baseline | Full Random Forest (notebook) | Light Random Forest (app) |
|---|---|---|---|
| Severity 4 recall | 0% | 55.9% | 52.3% |
| Severity macro F1 | 0.222 | 0.448 | 0.417 |
| Duration MAE (minutes) | 67.8 | 53.5 | 56.0 |

- Best model for both targets: Random Forest, chosen with 5-fold cross-validation.
- Severity 4 is missed more expensively than a false alarm, so the decision rule says
  "Severity 4" when its probability is above a threshold chosen on training data only.
- About 1 in 10 Severity 4 alerts is a real Severity 4, and closures longer than 4 hours are
  underestimated.
- Year is the strongest attribute. Without it the model is weaker (macro F1 0.439 to 0.375,
  MAE 57.8 to 64.3 minutes in the notebook) but still better than the baseline.
- The app uses a lighter Random Forest (50 trees, 100,000 rows) so it fits on GitHub and
  Streamlit Cloud.

## Reference

Moosavi, S., Samavatian, M. H., Parthasarathy, S., & Ramnath, R. (2019). A countrywide traffic
accident dataset. arXiv:1906.05409.
