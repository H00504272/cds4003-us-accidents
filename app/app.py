"""Streamlit app: predicts accident Severity (1-4) and clearing time (minutes).

Run from the repo root:  streamlit run app/app.py
"""
import datetime as dt
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'
CLASSES = np.array([1, 2, 3, 4])
POI = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Station',
       'Stop', 'Traffic_Calming', 'Traffic_Signal']
# same limits as the data preparation step: values outside these were treated as errors
LIMITS = {'Temperature(F)': (-60, 135), 'Humidity(%)': (0, 100), 'Pressure(in)': (20, 33),
          'Visibility(mi)': (0, 50), 'Wind_Speed(mph)': (0, 200), 'Precipitation(in)': (0, 25)}
SEV_TEXT = {1: 'Severity 1 - very short delay', 2: 'Severity 2 - short delay',
            3: 'Severity 3 - long delay', 4: 'Severity 4 - very long delay'}

st.set_page_config(page_title='US Accident Severity and Duration', layout='wide')


@st.cache_resource
def load():
    info = json.loads((MODELS / 'model_info.json').read_text())
    clf = joblib.load(MODELS / 'severity_model.joblib')
    reg = joblib.load(MODELS / 'duration_model.joblib')
    return info, clf, reg


try:
    info, clf, reg = load()
except Exception as e:  # for example a different scikit-learn version than the one used to train
    st.error('The model files could not be loaded. Train them again with '
             '`python src/train_light_model.py data/US_Accidents_March23_sampled_500k.csv`.')
    st.exception(e)
    st.stop()

m, rng, ch = info['metrics'], info['ranges'], info['choices']
THRESHOLD = info['threshold']

st.title('US accident severity and clearing time')
st.write('Enter what a traffic operator knows when an accident is reported. The app predicts '
         'how serious the traffic delay will be (Severity 1 to 4) and how many minutes the '
         'road will take to clear. It is a decision aid for a human dispatcher, not an '
         'automatic decision.')

# ------------------------------------------------------------------ inputs
with st.form('inputs'):
    a, b, c = st.columns(3)
    with a:
        st.subheader('When and where')
        date = st.date_input('Date', value=dt.date(2021, 1, 15),
                             min_value=dt.date(2016, 1, 1), max_value=dt.date(2023, 12, 31),
                             help='The data covers 2016 to 2023.')
        time = st.time_input('Time of the accident', value=dt.time(8, 30))
        state = st.selectbox('State', ch['State'], index=ch['State'].index(info['modes']['State']))
        tz = st.selectbox('Time zone', ch['Timezone'],
                          index=ch['Timezone'].index(info['modes']['Timezone']))
        lat = st.number_input('Latitude', value=34.05, format='%.4f',
                              help='Inside the United States: 24 to 50.')
        lng = st.number_input('Longitude', value=-118.24, format='%.4f',
                              help='Inside the United States: -125 to -66.')
        daylight = st.radio('Daylight', ['Day', 'Night'], horizontal=True)
    with b:
        st.subheader('Weather')
        weather = st.selectbox('Weather', ch['Weather_Group'],
                               index=ch['Weather_Group'].index(info['modes']['Weather_Group']))
        temp = st.number_input('Temperature (F)', value=60.0, step=1.0)
        hum = st.number_input('Humidity (%)', value=60.0, step=1.0)
        pres = st.number_input('Pressure (in)', value=29.9, step=0.1)
        vis = st.number_input('Visibility (miles)', value=10.0, step=0.5)
        wind = st.number_input('Wind speed (mph)', value=5.0, step=1.0)
        wdir = st.selectbox('Wind direction', ch['Wind_Direction'],
                            index=ch['Wind_Direction'].index(info['modes']['Wind_Direction']))
        precip = st.number_input('Precipitation (in)', value=0.0, step=0.01, format='%.2f')
    with c:
        st.subheader('Road features nearby')
        flags = {f: st.checkbox(f.replace('_', ' '), value=False) for f in POI}
    go = st.form_submit_button('Predict', type='primary')

if not go:
    st.info('Fill in the form and press Predict.')
    st.stop()

# ------------------------------------------------------------------ checks
errors = []
for name, val in {'Temperature(F)': temp, 'Humidity(%)': hum, 'Pressure(in)': pres,
                  'Visibility(mi)': vis, 'Wind_Speed(mph)': wind,
                  'Precipitation(in)': precip}.items():
    lo, hi = LIMITS[name]
    if not lo <= val <= hi:
        errors.append(f'{name} must be between {lo} and {hi} (you entered {val}).')
if not (24 <= lat <= 50 and -125 <= lng <= -66):
    errors.append('The location is outside the contiguous United States '
                  '(latitude 24 to 50, longitude -125 to -66).')
if errors:
    for e in errors:
        st.error(e)
    st.stop()

# ------------------------------------------------------------------ build the row
hour, month = time.hour, date.month
dow = date.strftime('%A')
season = ('Winter' if month in (12, 1, 2) else 'Spring' if month <= 5
          else 'Summer' if month <= 8 else 'Autumn')
row = {'Start_Lat': lat, 'Start_Lng': lng, 'Temperature(F)': temp, 'Humidity(%)': hum,
       'Pressure(in)': pres, 'Visibility(mi)': vis, 'Wind_Speed(mph)': wind,
       'POI_Count': sum(flags.values()), 'Hour': hour, 'Month': month, 'Year': date.year,
       'Precipitation(in)': precip,
       'Is_Weekend': int(dow in ('Saturday', 'Sunday')),
       'Is_RushHour': int(hour in (7, 8, 9, 16, 17, 18)),
       'State': state, 'Timezone': tz, 'Wind_Direction': wdir, 'Sunrise_Sunset': daylight,
       'Weather_Group': weather, 'Season': season, 'DayOfWeek': dow}
row.update({f: int(v) for f, v in flags.items()})
X = pd.DataFrame([row])[info['features']]


def predict(frame):
    proba = clf.predict_proba(frame)
    label = CLASSES[proba.argmax(axis=1)]
    label[proba[:, 3] >= THRESHOLD] = 4
    return proba, label, np.clip(reg.predict(frame), 5, 1440)


proba, label, minutes = predict(X)
p = proba[0]

# ------------------------------------------------------------------ result
st.header('Prediction')
r1, r2 = st.columns(2)
r1.metric('Predicted severity', SEV_TEXT[int(label[0])])
r2.metric('Predicted clearing time', f'{minutes[0]:.0f} min',
          help=f'About {minutes[0] / 60:.1f} hours. The typical error of the model is '
               f'{m["mae"]:.0f} minutes.')
if label[0] == 4:
    st.warning('Flagged as Severity 4. Suggested action: a dispatcher checks this accident '
               'before deciding on rerouting or sending extra crews.')
else:
    st.success('Not flagged as Severity 4. Normal handling.')

st.subheader('How sure is the model?')
st.bar_chart(pd.DataFrame({'probability': p}, index=[f'Severity {k}' for k in CLASSES]))
st.caption(f'The app says Severity 4 when its probability for class 4 is at least '
           f'{THRESHOLD:.2f}, even if another class is higher. A missed Severity 4 costs more '
           f'than a false alarm, so the rule is set to catch about half of them.')

# one-at-a-time test: put one input back to a typical value and see how much the answer moves
typical = {'Weather_Group': 'Clear', 'Visibility(mi)': rng['Visibility(mi)']['median'],
           'Precipitation(in)': 0.0, 'Wind_Speed(mph)': rng['Wind_Speed(mph)']['median'],
           'Temperature(F)': rng['Temperature(F)']['median'], 'Is_RushHour': 0, 'Is_Weekend': 0,
           'Sunrise_Sunset': 'Day', 'Traffic_Signal': 0, 'Junction': 0, 'Crossing': 0, 'Stop': 0}
rows = []
for feat, val in typical.items():
    if X.at[0, feat] == val:
        continue
    alt = X.copy()
    alt[feat] = val
    pa, _, ma = predict(alt)
    rows.append({'input': feat.replace('_', ' '), 'you entered': str(X.at[0, feat]),
                 'if typical value': str(val),
                 'change in Severity 4 probability (points)': round((p[3] - pa[0][3]) * 100, 1),
                 'change in minutes': round(float(minutes[0] - ma[0]), 1)})
st.subheader('What pushed this prediction')
if rows:
    st.write('Each row puts one of your inputs back to a typical value and shows how much the '
             'prediction would change. A positive number means your input made it higher.')
    st.dataframe(pd.DataFrame(rows).sort_values('change in minutes', key=abs, ascending=False),
                 hide_index=True)
else:
    st.write('All the inputs listed here are already at typical values.')

# ------------------------------------------------------------------ importance
st.header('What matters most for the model (all accidents)')
i1, i2 = st.columns(2)
with i1:
    st.write('**Severity** - drop in macro F1 when the attribute is shuffled')
    s = pd.Series(info['importance']['severity']).head(10)[::-1]
    st.bar_chart(s, horizontal=True)
with i2:
    st.write('**Duration** - extra minutes of error when the attribute is shuffled')
    d = pd.Series(info['importance']['duration']).head(10)[::-1]
    st.bar_chart(d, horizontal=True)
st.caption('Year is the strongest attribute for both targets. The data was collected over '
           'several years and its coverage changed, so Year partly measures how the data was '
           'collected, not only how accidents behave. The notebook shows the model without Year: '
           'it is weaker but still better than the baseline.')

# ------------------------------------------------------------------ business value
st.header('Business value and limits')
b1, b2, b3 = st.columns(3)
b1.metric('Severity 4 accidents caught', f'{m["sev4_recall"] * 100:.0f}%',
          help='Share of real Severity 4 accidents that the app flags (test set).')
b2.metric('Flagged alerts that are correct', f'{m["sev4_precision"] * 100:.0f}%',
          help='About 1 in 12 alerts is a real Severity 4. Most alerts are false alarms.')
b3.metric('Clearing time error vs simple guess',
          f'{(1 - m["mae"] / m["baseline_mae"]) * 100:.0f}% lower',
          help=f'Model error {m["mae"]:.0f} min against {m["baseline_mae"]:.0f} min when always '
               'guessing the median.')
st.write(
    f'On {m["test_rows"]:,} accidents the model never saw, the app catches about '
    f'{m["sev4_recall"] * 100:.0f}% of Severity 4 accidents and flags '
    f'{m["share_flagged_sev4"] * 100:.0f}% of all accidents to do it. Its macro F1 is '
    f'{m["macro_f1"]:.2f} against {m["baseline_macro_f1"]:.2f} for always guessing Severity 2. '
    'Because most alerts are false alarms, we recommend using it as a pilot with a human '
    'checking each alert, and not for automatic rerouting.')
st.caption(f'This app uses a lighter Random Forest (50 trees, {m["train_rows_used"]:,} training '
           'rows) so it fits on GitHub and Streamlit Cloud. The full model in the notebook is '
           'a little stronger. Closures longer than 4 hours are usually underestimated.')
