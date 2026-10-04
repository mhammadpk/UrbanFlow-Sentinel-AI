"""UrbanFlow Sentinel AI — hackathon-ready research demonstration.
Replace ONLY app.py in your GitHub repository. Existing artifacts remain unchanged.
Compatible with historical_predictions_h*.csv and predictions_h*.csv exports.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='UrbanFlow Sentinel AI | Smart Mobility', page_icon='🚦', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container{padding-top:1.5rem;max-width:1440px}h1,h2,h3{letter-spacing:-.025em}
.hero{padding:1.9rem 2rem;border-radius:18px;background:linear-gradient(115deg,#102b48,#0e6475);color:white;margin-bottom:1rem}
.hero h1{color:white;font-size:2.3rem;margin:0}.hero p{color:#e3f5f7;font-size:1.04rem;margin:.5rem 0}
.eyebrow{font-size:.78rem;font-weight:750;letter-spacing:.12em;color:#83e7d6;text-transform:uppercase}
.card{background:#f3f8fb;border:1px solid #dfeaf0;border-radius:14px;padding:1rem 1.2rem;margin:.4rem 0}
.card h4{margin:0 0 .35rem;color:#17415c}.card p{margin:0;color:#3f5767}
.step{padding:.8rem 1rem;border-left:4px solid #139c9d;background:#eff9f8;border-radius:8px;margin:.6rem 0}
[data-testid="stMetric"]{border:1px solid #e2eaf0;border-radius:12px;padding:1rem;background:#f8fafc}
footer{visibility:hidden}
</style>''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_metadata():
    p=BASE/'metadata.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}

meta=read_metadata()

def available_horizons():
    found=set()
    for pattern in ('historical_predictions_h*.csv','predictions_h*.csv'):
        for p in BASE.glob(pattern):
            try: found.add(int(p.stem.rsplit('h',1)[1]))
            except (ValueError,IndexError): pass
    return sorted(found)

H=available_horizons()
if not H:
    st.error('No historical forecast CSVs were found beside app.py. Upload historical_predictions_h1.csv (V3.2) or predictions_h1.csv (older export), plus the other horizons.')
    st.stop()

@st.cache_data(show_spinner=False)
def read_predictions(h):
    p=next((p for p in (BASE/f'historical_predictions_h{h}.csv', BASE/f'predictions_h{h}.csv') if p.exists()),None)
    df=pd.read_csv(p)
    aliases={'target':'actual','prediction':'predicted','lower_90':'lower','upper_90':'upper','date_time':'timestamp'}
    df=df.rename(columns={k:v for k,v in aliases.items() if k in df.columns and v not in df.columns})
    required={'timestamp','actual','predicted'}
    if not required.issubset(df.columns):
        raise ValueError(f'{p.name} lacks required columns: {sorted(required-set(df.columns))}')
    df['timestamp']=pd.to_datetime(df['timestamp'],errors='coerce')
    for c in ['actual','predicted','lower','upper']:
        if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
    df=df.dropna(subset=['timestamp','actual','predicted']).sort_values('timestamp').reset_index(drop=True)
    if 'lower' not in df: df['lower']=np.nan
    if 'upper' not in df: df['upper']=np.nan
    df['lower']=df['lower'].clip(lower=0)
    return df

@st.cache_data(show_spinner=False)
def optional_csv(name):
    for p in (BASE/name,BASE/'metrics'/name,BASE/'research'/name):
        if p.exists():
            try:return pd.read_csv(p)
            except Exception:return None
    return None

def horizon_label(h):
    minutes=meta.get('horizon_minutes',{}).get(str(h))
    if minutes is None: minutes=h*(5 if meta.get('data_mode')=='sensor_csv' else 60)
    return f'{int(minutes)} minutes ahead' if minutes<60 else f'{minutes/60:g} hour'+('s' if minutes!=60 else '')+' ahead'

st.sidebar.markdown('## 🚦 UrbanFlow Sentinel AI')
st.sidebar.caption('Smart mobility • Research prototype')
page=st.sidebar.radio('Explore', ['🏠 Overview','🧭 Guided Demo','📈 Forecast Explorer','🧠 AI Insights','📊 Research Evidence','ℹ️ About & Roadmap'],label_visibility='collapsed')
st.sidebar.divider()
st.sidebar.caption('DATA STATUS')
st.sidebar.info('Historical research dataset · Not live traffic')
st.sidebar.caption('The UCI study uses one monitoring location and measures traffic volume, not observed congestion.')

if page=='🏠 Overview':
    st.markdown('''<div class="hero"><div class="eyebrow">Smart mobility • Explainable AI</div><h1>See tomorrow’s traffic patterns, sooner.</h1><p>Explore how artificial intelligence can forecast traffic demand hours ahead, explain predictions, and support future proactive planning.</p></div>''',unsafe_allow_html=True)
    st.caption('Research demonstration using historical single-site traffic-volume data. Not a live or operational congestion-management system.')
    a,b,c=st.columns(3)
    with a: st.markdown('<div class="card"><h4>① Explore</h4><p>Select a historical traffic example without uploading any data.</p></div>',unsafe_allow_html=True)
    with b: st.markdown('<div class="card"><h4>② Understand</h4><p>Compare observed and forecast traffic volumes with uncertainty bounds.</p></div>',unsafe_allow_html=True)
    with c: st.markdown('<div class="card"><h4>③ Evaluate</h4><p>Inspect research accuracy, model comparisons and limitations.</p></div>',unsafe_allow_html=True)
    st.info('**Start here:** Select **🧭 Guided Demo** in the sidebar. Pick a forecast horizon, choose a historical example, and compare the prediction with what was later observed.')
    st.subheader('What does the prototype demonstrate?')
    x,y=st.columns([1.5,1])
    with x:
        st.write('UrbanFlow Sentinel AI forecasts **traffic volume (vehicles per hour)** at a monitored location using historical traffic patterns. A higher forecast indicates greater expected traffic demand, **not necessarily congestion**.')
        st.write('**For judges:** Open **Guided Demo** in the left navigation to experience a preloaded example in under a minute.')
    with y:
        st.info('**Research scope**\n\n✓ Hourly forecasting\n\n✓ Model evaluation\n\n✓ Uncertainty visualization\n\n✗ Live traffic feeds\n\n✗ Verified citywide hotspots')
    st.subheader('Preview: actual versus forecast')
    h=H[0]; df=read_predictions(h)
    if not df.empty:
        view=df.tail(min(72,len(df)))
        st.line_chart(view.set_index('timestamp')[['actual','predicted']],height=270)
        st.caption(f'Last {len(view)} saved held-out observations · {horizon_label(h)}. This is historical evaluation, not a live prediction.')

elif page=='🧭 Guided Demo':
    st.title('🧭 Guided Demo')
    st.write('A complete demonstration using saved historical predictions—**no technical input or CSV required**.')
    h=st.select_slider('Step 1 · How far ahead should the system forecast?',options=H,value=H[0],format_func=horizon_label)
    df=read_predictions(h)
    if df.empty: st.warning('No usable saved predictions.');st.stop()
    n=min(len(df),240)
    examples=df.tail(n).reset_index(drop=True)
    i=st.selectbox(
        'Step 2 · Choose a historical example',
        options=list(range(n)),
        index=min(24,n-1),
        format_func=lambda j: examples.iloc[int(j)]['timestamp'].strftime('%d %b %Y, %H:%M'),
        help='Select a date and time from the held-out historical test period.',
    )
    r=examples.iloc[int(i)]
    st.markdown('<div class="step"><strong>Step 3 · Read the forecast</strong><br>Compare the AI prediction with the value observed later in the historical record.</div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric('AI forecast',f"{r['predicted']:,.0f}",help='Estimated traffic volume in vehicles per hour')
    b.metric('Observed later',f"{r['actual']:,.0f}",help='Actual traffic volume from the held-out historical dataset')
    c.metric('Absolute difference',f"{abs(r['predicted']-r['actual']):,.0f}",help='Absolute forecast error for this example')
    if pd.notna(r['lower']) and pd.notna(r['upper']):
        st.info(f"**Prediction interval:** {r['lower']:,.0f}–{r['upper']:,.0f} vehicles/hour. This is an empirical uncertainty range, not a guarantee.")
    row_idx=len(df)-n+int(i)
    context=df.iloc[max(0,row_idx-12):min(len(df),row_idx+13)].copy()
    st.line_chart(context.set_index('timestamp')[['actual','predicted']],height=330)
    difference=float(r['predicted']-r['actual'])
    if difference>0:
        interpretation=f'The model forecast about {abs(difference):,.0f} more vehicles/hour than were subsequently observed.'
    elif difference<0:
        interpretation=f'The model forecast about {abs(difference):,.0f} fewer vehicles/hour than were subsequently observed.'
    else:
        interpretation='The predicted and observed traffic volumes match for this example.'
    st.success(f'**Plain-language explanation:** {interpretation} Traffic volume is not the same as traffic congestion.')
    with st.expander('What do the numbers mean?'):
        st.markdown('**AI forecast:** Estimated vehicles passing the sensor in the target hour.\n\n**Observed later:** Recorded vehicle count for that hour.\n\n**Absolute difference:** The gap between the prediction and the observed count.\n\n**Prediction interval:** A range derived from historical model errors; it is not guaranteed to contain every future observation.')
    st.caption('All examples come from previously saved held-out research results. Selecting an example does not retrain the model.')

elif page=='📈 Forecast Explorer':
    st.title('📈 Forecast Explorer')
    st.write('Explore real held-out research predictions and compare them with observed traffic volume.')
    h=st.selectbox('Forecast horizon',H,format_func=horizon_label)
    df=read_predictions(h)
    if 'sensor_id' in df.columns and df.sensor_id.nunique()>1:
        sensor=st.selectbox('Monitoring location',sorted(df.sensor_id.dropna().astype(str).unique()))
        df=df[df.sensor_id.astype(str)==sensor]
    if df.empty: st.warning('No records available.');st.stop()
    max_rows=min(len(df),336)
    window=st.slider('Number of recent historical observations',min(24,max_rows),max_rows,min(96,max_rows),step=1) if max_rows>=24 else max_rows
    view=df.tail(window).copy()
    a,b,c=st.columns(3)
    a.metric('Historical observations',f'{len(df):,}')
    b.metric('MAE · selected window',f"{(view.actual-view.predicted).abs().mean():,.1f}",help='Average absolute prediction error, in vehicles/hour')
    c.metric('Forecast horizon',horizon_label(h))
    st.subheader('Observed vs predicted traffic volume')
    st.caption('Horizontal axis: historical date/time · Vertical axis: vehicles per hour. Closer lines indicate more accurate forecasts.')
    st.line_chart(view.set_index('timestamp')[['actual','predicted']],height=360)
    if view[['lower','upper']].notna().all().all():
        with st.expander('Show uncertainty bounds',expanded=False):
            st.line_chart(view.set_index('timestamp')[['lower','predicted','upper']],height=250)
            st.caption('Prediction intervals are empirical. They may under-cover in particular months or changing conditions.')
    with st.expander('Inspect and download historical records'):
        st.dataframe(view[['timestamp','actual','predicted','lower','upper']].round(1),use_container_width=True,hide_index=True)
        st.download_button('Download selected records',view.to_csv(index=False).encode(),f'urbanflow_forecasts_{h}h.csv','text/csv')
    st.caption('Traffic volume is a demand measure. This chart does not represent measured speed, delays, or congestion hotspots.')

elif page=='🧠 AI Insights':
    st.title('🧠 AI Insights')
    st.write('Discover which input patterns the AI model relies on most. Larger SHAP bars indicate stronger average influence on the model output—not causes of congestion.')
    h=st.selectbox('Explain forecast horizon',H,format_func=horizon_label)
    shap=optional_csv(f'shap_importance_h{h}.csv')
    if shap is not None and len(shap):
        st.subheader('Global feature importance (SHAP)')
        num=[c for c in shap.columns if pd.api.types.is_numeric_dtype(shap[c])]
        txt=[c for c in shap.columns if c not in num]
        if num and txt:
            value_col=next((c for c in num if 'mean' in c.lower() or 'importance' in c.lower()),num[0])
            label_col=next((c for c in txt if 'feature' in c.lower()),txt[0])
            friendly={'origin_value':'Traffic at forecast origin','hour':'Hour of day','dow':'Day of week','month':'Month','is_weekend':'Weekend indicator',
                      'rolling_mean_3':'Recent 3-hour average','rolling_mean_24':'Recent 24-hour average','rolling_mean_168':'Recent 7-day average'}
            def friendly_feature(raw):
                name=str(raw).replace('num__','').replace('cat__','')
                if name.startswith('lag_'):
                    return f'Traffic {name.split("_")[-1]} hours earlier'
                return friendly.get(name,name.replace('_',' ').capitalize())
            view=shap[[label_col,value_col]].dropna().sort_values(value_col,ascending=False).head(10).copy()
            view['friendly_name']=view[label_col].map(friendly_feature)
            st.bar_chart(view.set_index('friendly_name')[value_col],horizontal=True)
            with st.expander('Show original SHAP feature names'):
                st.dataframe(view[[label_col,'friendly_name',value_col]],hide_index=True,use_container_width=True)
            st.caption('Global SHAP importance summarizes average model sensitivity; it is not a causal explanation or a local explanation for a selected example.')
        else: st.dataframe(shap,use_container_width=True)
    else:
        st.info('Detailed SHAP CSVs are not included in this deployment. Copy shap_importance_h1.csv, shap_importance_h3.csv and shap_importance_h6.csv from your frozen experiment metrics folder into a `research/` folder to enable the charts.')
    st.subheader('How to interpret model inputs')
    for title,desc in [('Recent traffic history','Lagged traffic-volume measurements help the model recognize current demand.'),('Time of day and day of week','Recurring commuting and daily activity patterns help contextualize forecasts.'),('Rolling traffic patterns','Recent averages can smooth noisy observations, but missing history may reduce reliability.')]:
        st.markdown(f'<div class="card"><h4>{title}</h4><p>{desc}</p></div>',unsafe_allow_html=True)
    st.warning('Feature importance describes model behavior. It does not prove that changing any factor would reduce congestion.')

elif page=='📊 Research Evidence':
    st.title('📊 Research Evidence')
    st.write('How well did the models predict historical traffic? Lower forecasting error is better. These are held-out research results, not live performance.')
    comp=optional_csv('model_comparison.csv')
    if comp is not None:
        comp=comp.rename(columns={c:c.strip().lower() for c in comp.columns})
    if comp is not None and {'model','mae'}.issubset(comp.columns):
        hcol='horizon_steps' if 'horizon_steps' in comp.columns else ('horizon' if 'horizon' in comp.columns else None)
        if hcol:
            choices=sorted(comp[hcol].dropna().unique().tolist())
            selected=st.selectbox('Evaluation horizon',choices,format_func=lambda x:horizon_label(int(x)))
            part=comp[comp[hcol]==selected].copy()
        else: part=comp.copy()
        part=part.sort_values('mae')
        best=part.iloc[0]
        st.metric('Lowest MAE on this horizon',f"{best['mae']:,.1f} vehicles/hour",help=f"Model: {best['model']}")
        st.caption('MAE means the average size of the forecast error. For example, MAE = 150 means forecasts differed from the recorded hourly volume by about 150 vehicles on average.')
        st.subheader('Mean absolute error (lower is better)')
        st.bar_chart(part.set_index('model')['mae'],horizontal=True,height=290)
        display=[c for c in ['model','mae','rmse','r2','coverage_90','mean_interval_width','n_test'] if c in part]
        st.dataframe(part[display].round(3),hide_index=True,use_container_width=True)
        st.caption('MAE and RMSE are in vehicles/hour. R² describes explained variation on the evaluated sample. Results vary by study version.')
    else: st.warning('Model comparison is unavailable. Check that model_comparison.csv is committed and contains model and MAE columns.')
    for filename,title in [('rolling_origin.csv','Rolling-origin validation'),('paired_bootstrap.csv','Paired statistical comparisons'),('conformal.csv','Prediction-interval calibration'),('missing_history_sensitivity.csv','Missing-history sensitivity'),('monthly_coverage.csv','Monthly interval coverage'),('subgroups.csv','Subgroup evaluation')]:
        data=optional_csv(filename)
        if data is not None:
            with st.expander(title):
                st.dataframe(data,use_container_width=True,hide_index=True)
                st.download_button(f'Download {title.lower()} CSV',data.to_csv(index=False).encode('utf-8'),filename,'text/csv',key=f'download_{filename}')
    st.info('**Scientific interpretation:** High accuracy on a single-site historical traffic-volume dataset does not establish accurate congestion prediction, transferability to other cities, or effectiveness of interventions.')

else:
    st.title('ℹ️ About & Roadmap')
    st.subheader('Research demonstration')
    st.write('UrbanFlow Sentinel AI is an explainable traffic-demand forecasting research prototype. It illustrates how historical traffic-volume observations can support short-term prediction and uncertainty-aware decision support.')
    st.subheader('How it works')
    a,b,c,d=st.columns(4)
    for col,emoji,title,detail in [(a,'📥','Historical data','Hourly traffic observations'),(b,'🧹','Quality checks','Missingness and temporal features'),(c,'🤖','AI forecast','Models estimate future volume'),(d,'📊','Interpret','Forecast, uncertainty and validation')]:
        with col:
            st.markdown(f'<div class="card"><h4>{emoji} {title}</h4><p>{detail}</p></div>',unsafe_allow_html=True)
    st.subheader('What is available now')
    st.success('Historical forecasts, evaluation charts, guided demonstration, and downloadable historical records.')
    st.subheader('Future development — not yet validated')
    st.markdown('Multi-sensor traffic speed and occupancy data; geolocated hotspot forecasting; shorter forecasting horizons; operational data integration; evaluated intervention scenarios; Saudi-city pilot validation.')
    st.subheader('Data and responsible-use notes')
    st.write(f"**Dataset:** {meta.get('source','Historical research traffic-volume dataset')}")
    st.write(f"**Outcome:** {meta.get('outcome_interpretation','Traffic volume, not measured congestion')}")
    st.write('**Research limitations:** single-site data where applicable; empirical uncertainty intervals may vary across time; no causal intervention validation.')
    with st.expander('Technical provenance'):
        st.json(meta)

st.divider()
st.caption('UrbanFlow Sentinel AI · Research demonstration · Historical observations only · No operational traffic-control recommendations')
