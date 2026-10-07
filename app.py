"""UrbanFlow Sentinel AI — hackathon-ready research demonstration.
Replace ONLY app.py in your GitHub repository. Existing artifacts remain unchanged.
Compatible with historical_predictions_h*.csv and predictions_h*.csv exports.
"""
from __future__ import annotations
import json
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='UrbanFlow Sentinel AI | Smart Mobility', page_icon='🚦', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container{padding-top:3rem;max-width:1440px}h1,h2,h3{letter-spacing:-.025em}
.hero{padding:1.9rem 2rem;border-radius:18px;background:linear-gradient(115deg,#102b48,#0e6475);color:white;margin-bottom:1rem}
.hero h1{color:white;font-size:2.3rem;margin:0}.hero p{color:#e3f5f7;font-size:1.04rem;margin:.5rem 0}
.eyebrow{font-size:.78rem;font-weight:750;letter-spacing:.12em;color:#83e7d6;text-transform:uppercase}
.card{background:#f3f8fb;border:1px solid #dfeaf0;border-radius:14px;padding:1rem 1.2rem;margin:.4rem 0}
.card h4{margin:0 0 .35rem;color:#17415c}.card p{margin:0;color:#3f5767}
.step{padding:.8rem 1rem;border-left:4px solid #139c9d;background:#eff9f8;border-radius:8px;margin:.6rem 0}
[data-testid="stMetric"]{border:1px solid #e2eaf0;border-radius:12px;padding:1rem;background:#f8fafc}
.fc-brand{display:flex;align-items:center;gap:16px;padding:12px 16px;margin:0 0 1rem 0;border:1px solid #dfeaf0;border-radius:14px;background:#f8fbfd}
.fc-brand img{width:155px;max-width:100%;height:auto;display:block}
.fc-brand .label{font-size:.78rem;color:#607787;margin-bottom:2px}
.fc-brand .name{font-size:1.02rem;font-weight:750;color:#17415c;line-height:1.45}
.fc-brand .org{font-size:.82rem;color:#607787;margin-top:3px}
.fc-brand a{text-decoration:none}
@media(max-width:700px){.fc-brand{align-items:flex-start}.fc-brand img{width:120px}.fc-brand .name{font-size:.92rem}}
footer{visibility:hidden}
</style>''', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_metadata():
    p=BASE/'metadata.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}

meta=read_metadata()

def firstcity_research_branding():
    """Clickable First City RDI Center branding shown across the app."""
    logo_path = BASE / 'firstcity_logo.png'
    if not logo_path.exists():
        return
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode('ascii')
    st.markdown(
        f"""
        <div class="fc-brand">
          <a href="https://firstcity.sa/ar/research" target="_blank" rel="noopener noreferrer">
            <img src="data:image/png;base64,{logo_b64}" alt="First City">
          </a>
          <div>
            <a href="https://firstcity.sa/ar/research" target="_blank" rel="noopener noreferrer">
              <div class="name">Research, Development &amp; Innovation Center<br>مركز البحث والتطوير والابتكار</div>
            </a>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    """Find metrics in either repo root, metrics/, or research/; don't silently swallow errors."""
    for p in (BASE/name, BASE/'metrics'/name, BASE/'research'/name):
        if p.is_file():
            try:
                df=pd.read_csv(p, encoding='utf-8-sig')
                df.columns=df.columns.astype(str).str.strip()
                return df
            except Exception as exc:
                st.warning(f'Could not read {p.relative_to(BASE)} ({type(exc).__name__}). Please check its CSV formatting.')
    return None

def normalized_comparison():
    df=optional_csv('model_comparison.csv')
    if df is None or df.empty:
        return None
    df=df.rename(columns={c:c.strip().lower() for c in df.columns})
    if not {'model','mae'}.issubset(df.columns):
        return None
    if 'horizon_steps' not in df.columns and 'horizon' in df.columns:
        df=df.rename(columns={'horizon':'horizon_steps'})
    if 'horizon_steps' not in df.columns:
        return None
    df['horizon_steps']=pd.to_numeric(df['horizon_steps'],errors='coerce')
    for col in ('mae','rmse','r2'):
        if col in df: df[col]=pd.to_numeric(df[col],errors='coerce')
    return df.dropna(subset=['horizon_steps','mae','model'])

def model_name(name):
    return {'rf':'Random Forest','lgbm':'LightGBM','historical_profile':'Historical profile',
            'daily_naive':'Previous day','weekly_naive':'Previous week','persistence':'Current traffic',
            'ridge':'Ridge regression'}.get(str(name),str(name).replace('_',' ').title())

def metric_row(h,model):
    comp=normalized_comparison()
    if comp is None:return None
    rows=comp[(comp.horizon_steps==h)&(comp.model.astype(str).str.lower()==model)]
    return rows.iloc[0] if not rows.empty else None

def percent_improvement(h):
    base=metric_row(h,'historical_profile'); rf=metric_row(h,'rf')
    if base is None or rf is None or float(base.mae)<=0:return None
    return 100*(float(base.mae)-float(rf.mae))/float(base.mae)

def safe_feature_name(raw):
    name=str(raw).replace('num__','').replace('cat__','')
    labels={'origin_value':'Traffic at forecast origin','hour':'Hour of day','dow':'Day of week',
            'month':'Month of year','is_weekend':'Weekend','rolling_mean_3':'Recent 3-hour average',
            'rolling_mean_24':'Recent 24-hour average','rolling_mean_168':'Recent 7-day average'}
    if name.startswith('lag_'):
        return f'Traffic {name.split("_")[-1]} hours earlier'
    return labels.get(name,name.replace('_',' ').capitalize())


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

firstcity_research_branding()

if page=='🏠 Overview':
    st.markdown('''<div class="hero"><div class="eyebrow">Smart mobility • Explainable AI</div><h1>See tomorrow’s traffic patterns, sooner.</h1><p>Explore how artificial intelligence can forecast traffic demand hours ahead, explain predictions, and support future proactive planning.</p></div>''',unsafe_allow_html=True)
    st.caption('Research demonstration using historical single-site traffic-volume data. Not a live or operational congestion-management system.')
    a,b,c=st.columns(3)
    with a: st.markdown('<div class="card"><h4>① Explore</h4><p>Select a historical traffic example without uploading any data.</p></div>',unsafe_allow_html=True)
    with b: st.markdown('<div class="card"><h4>② Understand</h4><p>Compare observed and forecast traffic volumes with uncertainty bounds.</p></div>',unsafe_allow_html=True)
    with c: st.markdown('<div class="card"><h4>③ Evaluate</h4><p>Inspect research accuracy, model comparisons and limitations.</p></div>',unsafe_allow_html=True)
    st.info('**Start here:** Select **🧭 Guided Demo** in the sidebar. Pick a forecast horizon, choose a historical example, and compare the prediction with what was later observed.')
    st.subheader('Research results at a glance')
    comparison=normalized_comparison()
    if comparison is not None:
        first=metric_row(1,'rf')
        imp=percent_improvement(1)
        k1,k2,k3,k4=st.columns(4)
        k1.metric('Forecast horizons',f'{len(H)}',help='Separate 1-, 3- and 6-hour prediction tasks.')
        k2.metric('Best 1-hour test MAE',f'{float(first.mae):,.1f}' if first is not None else 'N/A',help='Random Forest average absolute error, vehicles/hour, on the held-out test set.')
        k3.metric('Error reduction vs historical profile',f'{imp:.1f}%' if imp is not None else 'N/A',help='Relative reduction in test MAE for Random Forest at 1 hour; not a reduction in congestion.')
        k4.metric('Monitoring locations','1' if meta.get('data_mode','uci')=='uci' else 'Dataset-defined',help='UCI research data use one traffic-count monitoring location.')
        st.caption('Metrics are read from model_comparison.csv in your repository. The live demo uses saved LightGBM forecasts; Random Forest achieved the lowest MAE in the recorded comparisons.')
    else:
        st.warning('Research metrics are not available. Confirm model_comparison.csv is beside app.py or inside research/.')
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
    if shap is not None and not shap.empty:
        shap=shap.rename(columns={c:c.strip().lower() for c in shap.columns})
        if {'feature','mean_abs_shap'}.issubset(shap.columns):
            shap['mean_abs_shap']=pd.to_numeric(shap['mean_abs_shap'],errors='coerce')
            view=shap.dropna(subset=['feature','mean_abs_shap']).sort_values('mean_abs_shap',ascending=False).head(8).copy()
            view['friendly_name']=view['feature'].map(safe_feature_name)
            if not view.empty:
                st.subheader('Which signals matter most?')
                st.bar_chart(view.set_index('friendly_name')['mean_abs_shap'].sort_values(),horizontal=True,height=320)
                top=view.iloc[0]
                st.info(f"**In simple terms:** {top['friendly_name']} was the strongest average influence on the {horizon_label(h)} model in the saved SHAP sample. This explains model sensitivity, not the cause of traffic congestion.")
                with st.expander('Technical SHAP data and original feature names'):
                    st.dataframe(view[['feature','friendly_name','mean_abs_shap']],hide_index=True,use_container_width=True)
            else:st.warning('The selected SHAP CSV has no valid importance values.')
        else:
            st.warning(f'The h{h} SHAP file is present but lacks feature and mean_abs_shap columns.')
    else:
        st.warning(f'SHAP results for {horizon_label(h)} could not be loaded. Verify research/shap_importance_h{h}.csv exists on the GitHub main branch.')
        with st.expander('Troubleshooting: expected paths'):
            st.code('\n'.join(str(p.relative_to(BASE)) for p in [BASE/f'shap_importance_h{h}.csv',BASE/'metrics'/f'shap_importance_h{h}.csv',BASE/'research'/f'shap_importance_h{h}.csv']))
    st.subheader('How to interpret model inputs')
    for title,desc in [('Recent traffic history','Lagged traffic-volume measurements help the model recognize current demand.'),('Time of day and day of week','Recurring commuting and daily activity patterns help contextualize forecasts.'),('Rolling traffic patterns','Recent averages can smooth noisy observations, but missing history may reduce reliability.')]:
        st.markdown(f'<div class="card"><h4>{title}</h4><p>{desc}</p></div>',unsafe_allow_html=True)
    st.warning('Feature importance describes model behavior. It does not prove that changing any factor would reduce congestion.')

elif page=='📊 Research Evidence':
    st.title('📊 Research Evidence')
    st.write('How well did the models predict historical traffic? Lower forecasting error is better. These are held-out research results, not live performance.')
    comp=normalized_comparison()
    if comp is not None and not comp.empty:
        horizons=sorted(set(int(x) for x in comp.horizon_steps.dropna()))
        selected=st.selectbox('Choose forecast horizon',horizons,format_func=horizon_label)
        part=comp[comp.horizon_steps==selected].copy().sort_values('mae')
        best=part.iloc[0]
        rf=metric_row(selected,'rf'); lgbm=metric_row(selected,'lgbm'); profile=metric_row(selected,'historical_profile')
        a,b,c,d=st.columns(4)
        a.metric('Best model',model_name(best['model']))
        b.metric('Best MAE',f"{best['mae']:,.1f}",help='Average absolute error in vehicles/hour; lower is better.')
        c.metric('LightGBM R²',f"{float(lgbm['r2']):.3f}" if lgbm is not None and 'r2' in lgbm else 'N/A',help='Fraction of test-set variation explained by the LightGBM forecasts.')
        gain=percent_improvement(selected)
        d.metric('RF error reduction vs profile',f'{gain:.1f}%' if gain is not None else 'N/A',help='Reduction in MAE compared with the historical-profile baseline, not a reduction in traffic or congestion.')
        st.subheader('Model comparison — average forecast error')
        plot=part[['model','mae']].copy()
        plot['model']=plot['model'].map(model_name)
        st.bar_chart(plot.set_index('model')['mae'].sort_values(),horizontal=True,height=340)
        st.caption('Each bar shows mean absolute error (MAE) on the saved held-out test data, measured in vehicles/hour. Shorter bars are better.')
        if rf is not None and lgbm is not None:
            st.info(f"**What the evidence says:** Random Forest has MAE {rf.mae:,.1f}; the deployed LightGBM model has MAE {lgbm.mae:,.1f} for {horizon_label(selected)}. The deployment uses LightGBM, not the lowest-MAE model. This distinction is intentional and transparent.")
        with st.expander('View full model comparison and download'):
            display=part.copy();display['model']=display['model'].map(model_name)
            st.dataframe(display.round(3),hide_index=True,use_container_width=True)
            st.download_button('Download model comparison',comp.to_csv(index=False).encode(),'model_comparison.csv','text/csv')
        st.subheader('Reliability and validation')
        conf=optional_csv('conformal.csv')
        if conf is not None:
            conf.columns=conf.columns.str.strip().str.lower()
            if {'horizon_steps','model','coverage'}.issubset(conf.columns):
                cc=conf[(pd.to_numeric(conf.horizon_steps,errors='coerce')==selected)&(conf.model.astype(str).str.lower()=='lgbm')]
                if not cc.empty:
                    coverage=float(cc.iloc[0]['coverage'])*100
                    st.metric('Observed interval coverage (LightGBM)',f'{coverage:.1f}%',help='Share of test observations within the empirical prediction interval. Nominal target is 90%; actual coverage varies by month.')
                    st.caption('The prediction interval has a nominal 90% target; empirical coverage is not guaranteed in every month.')
        rolling=optional_csv('rolling_origin.csv')
        if rolling is not None:
            rolling=rolling.rename(columns={c:c.strip().lower() for c in rolling.columns})
            if {'fold','horizon_steps','model','mae'}.issubset(rolling.columns):
                folds=rolling[(pd.to_numeric(rolling.horizon_steps,errors='coerce')==selected)&(rolling.model.astype(str).str.lower().isin(['rf','lgbm']))]
                if not folds.empty:
                    st.markdown('**Rolling-origin validation** — how error changes across different chronological evaluation folds')
                    st.line_chart(folds.pivot_table(index='fold',columns='model',values='mae'),height=240)
        with st.expander('Additional statistical and robustness results'):
            for filename,title in [('paired_bootstrap.csv','Paired model comparisons'),('missing_history_sensitivity.csv','Missing-history sensitivity'),('monthly_coverage.csv','Monthly uncertainty coverage'),('subgroups.csv','Peak/off-peak analysis')]:
                data=optional_csv(filename)
                if data is not None:
                    st.markdown(f'**{title}**')
                    st.dataframe(data,hide_index=True,use_container_width=True)
    else:
        st.error('Model comparison cannot be displayed. Please verify model_comparison.csv exists in the GitHub repository and includes horizon_steps, model, and MAE columns.')
        with st.expander('Diagnostic — file search'):
            st.write('App folder:',str(BASE))
            st.write('Candidate files:',[str(p) for p in (BASE/'model_comparison.csv',BASE/'metrics/model_comparison.csv',BASE/'research/model_comparison.csv')])
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
    st.markdown('**Developed by:** [Research, Development & Innovation Center | مركز البحث والتطوير والابتكار](https://firstcity.sa/ar/research) · First City for Information Technology')
    with st.expander('Technical provenance'):
        st.json(meta)

st.divider()
st.caption('UrbanFlow Sentinel AI · Research demonstration · Historical observations only · No operational traffic-control recommendations')
