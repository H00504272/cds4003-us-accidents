"""Train the small models used by the Streamlit app.

The notebooks use Random Forest on all 400k training rows. That model is too big for GitHub
(100 MB limit) and for Streamlit Community Cloud, so the app uses a lighter Random Forest:
fewer trees, shallower trees, and a 100k-row training sample. The test set is the same
20% split as in the CLO3 notebook (seed 42), so the numbers are comparable.

Run from the repo root:  python src/train_light_model.py data/US_Accidents_March23_sampled_500k.csv
"""
import json
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, mean_absolute_error, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, 'src')
from prep import prepare

SEED = 42
NUM = ['Start_Lat', 'Start_Lng', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
       'Visibility(mi)', 'Wind_Speed(mph)', 'POI_Count', 'Hour', 'Month', 'Year']
PRECIP = ['Precipitation(in)']
FLAGS = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Station',
         'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Is_Weekend', 'Is_RushHour']
CAT = ['State', 'Timezone', 'Wind_Direction', 'Sunrise_Sunset', 'Weather_Group', 'Season',
       'DayOfWeek']
FEATURES = NUM + PRECIP + FLAGS + CAT

path = sys.argv[1]
_, df = prepare(path)                      # the table with missing values left empty
X = df[FEATURES].copy()
X[FLAGS] = X[FLAGS].astype(int)
y_sev, y_dur = df['Severity'], df['Duration_Minutes']

X_train, X_test, ys_train, ys_test, yd_train, yd_test = train_test_split(
    X, y_sev, y_dur, test_size=0.2, stratify=y_sev, random_state=SEED)
# 100k rows to fit, 50k other training rows to choose the threshold
X_fit, X_rest, ys_fit, ys_rest, yd_fit, yd_rest = train_test_split(
    X_train, ys_train, yd_train, train_size=100000, stratify=ys_train, random_state=SEED)
X_val, ys_val = X_rest.head(50000), ys_rest.head(50000)

preprocess = ColumnTransformer([
    ('num', Pipeline([('fill', SimpleImputer(strategy='mean')), ('scale', StandardScaler())]), NUM),
    ('precip', Pipeline([('fill', SimpleImputer(strategy='constant', fill_value=0)),
                         ('scale', StandardScaler())]), PRECIP),
    ('flags', 'passthrough', FLAGS),
    ('cat', Pipeline([('fill', SimpleImputer(strategy='most_frequent')),
                      ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
     CAT)])

clf = Pipeline([('prep', preprocess),
                ('model', RandomForestClassifier(n_estimators=50, max_depth=18, min_samples_leaf=3,
                                                 max_features=0.3, class_weight='balanced',
                                                 n_jobs=-1, random_state=SEED))])
reg = Pipeline([('prep', preprocess),
                ('model', TransformedTargetRegressor(
                    regressor=RandomForestRegressor(n_estimators=50, max_depth=18,
                                                    min_samples_leaf=3, n_jobs=-1,
                                                    random_state=SEED),
                    func=np.log1p, inverse_func=np.expm1))])
clf.fit(X_fit, ys_fit)
reg.fit(X_fit, yd_fit)

classes = np.array([1, 2, 3, 4])


def apply_rule(proba, t):
    pred = classes[proba.argmax(axis=1)]
    pred[proba[:, 3] >= t] = 4
    return pred


# threshold: highest value that still catches at least 50% of Severity 4 on the validation rows
p_val = clf.predict_proba(X_val)
best_t = 0.05
for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
    if recall_score(ys_val, apply_rule(p_val, t), labels=[4], average='macro') >= 0.5:
        best_t = float(t)
print('threshold:', best_t)

p_test = apply_rule(clf.predict_proba(X_test), best_t)
d_pred = reg.predict(X_test)
base_mae = float(np.abs(yd_test - yd_train.median()).mean())
metrics = {
    'sev4_recall': float(recall_score(ys_test, p_test, labels=[4], average='macro')),
    'sev4_precision': float(precision_score(ys_test, p_test, labels=[4], average='macro')),
    'macro_f1': float(f1_score(ys_test, p_test, average='macro')),
    'baseline_macro_f1': float(f1_score(ys_test, np.full(len(ys_test), 2), average='macro')),
    'mae': float(mean_absolute_error(yd_test, d_pred)),
    'baseline_mae': base_mae,
    'test_rows': int(len(X_test)), 'train_rows_used': int(len(X_fit)),
    'share_flagged_sev4': float((p_test == 4).mean()),
}
print(metrics)

# which attributes matter most (permutation importance on 10k test rows)
sub = X_test.sample(10000, random_state=SEED)
imp_c = permutation_importance(clf, sub, ys_test.loc[sub.index], scoring='f1_macro',
                               n_repeats=3, random_state=SEED, n_jobs=1)
imp_r = permutation_importance(reg, sub, yd_test.loc[sub.index],
                               scoring='neg_mean_absolute_error', n_repeats=3,
                               random_state=SEED, n_jobs=1)
importance = {
    'severity': dict(sorted(zip(FEATURES, imp_c.importances_mean.round(4).tolist()),
                            key=lambda kv: -kv[1])),
    'duration': dict(sorted(zip(FEATURES, imp_r.importances_mean.round(3).tolist()),
                            key=lambda kv: -kv[1])),
}

# allowed values and ranges for the input checks and defaults in the app
ranges = {c: {'min': float(X_train[c].min()), 'max': float(X_train[c].max()),
              'median': float(X_train[c].median())} for c in NUM + PRECIP}
choices = {c: sorted(X_train[c].dropna().unique().tolist()) for c in CAT}
modes = {c: X_train[c].mode().iloc[0] for c in CAT}
flag_rates = {c: float(X_train[c].mean()) for c in FLAGS}
info = {'threshold': best_t, 'metrics': metrics, 'importance': importance, 'ranges': ranges,
        'choices': choices, 'modes': modes, 'flag_rates': flag_rates, 'features': FEATURES,
        'severity_share_train': (ys_train.value_counts(normalize=True).sort_index()
                                 .round(4).to_dict()),
        'duration_median_train': float(yd_train.median())}
with open('models/model_info.json', 'w') as f:
    json.dump(info, f, indent=1, default=str)
joblib.dump(clf, 'models/severity_model.joblib', compress=3)
joblib.dump(reg, 'models/duration_model.joblib', compress=3)
print('saved')
