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
    labels={'origin_value':'Latest observed traffic volume','hour':'Hour of day','dow':'Day of week',
            'month':'Month of year','is_weekend':'Weekend','rolling_mean_3':'Recent 3-hour average',
            'rolling_mean_24':'Recent 24-hour average','rolling_mean_168':'Recent 7-day average'}
    if name.startswith("lag_"):
        hours = int(name.split("_")[-1])
        unit = "hour" if hours == 1 else "hours"
        return f"Traffic {hours} {unit} earlier"
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
        demo_h = 1 if 1 in H else H[0]
        demo_model = metric_row(demo_h, 'lgbm')
        research_best = comparison[comparison.horizon_steps == demo_h].sort_values('mae')
        research_best = research_best.iloc[0] if not research_best.empty else None
        k1,k2,k3,k4=st.columns(4)
        k1.metric('Demo forecasting model', 'LightGBM',
                  help='Saved LightGBM predictions power the interactive demonstration.')
        k2.metric(f'LightGBM test MAE · {horizon_label(demo_h)}',
                  f'{float(demo_model.mae):,.1f} vehicles/hour' if demo_model is not None else 'N/A',
                  help='Average absolute prediction error on the held-out historical test set; lower is better.')
        k3.metric('Forecast horizons', f'{len(H)}',
                  help='Available saved forecast horizons.')
        k4.metric('Monitoring locations',
                  '1' if meta.get('data_mode','uci')=='uci' else 'Dataset-defined',
                  help='The UCI research data use one traffic-count monitoring location.')
        if research_best is not None and str(research_best['model']).lower() != 'lgbm':
            st.caption(
                f'**Demo versus research comparison:** LightGBM powers the saved forecasts. '
                f'{model_name(research_best["model"])} achieved the lowest overall test error '
                f'for {horizon_label(demo_h)} '
                f'({float(research_best["mae"]):,.1f} vehicles/hour). '
                'See Research Evidence for the full comparison.'
            )
        else:
            st.caption('Demo forecasts use LightGBM. Research Evidence contains the full held-out model comparison.')
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
                st.subheader("What information influences the AI forecast?")

                st.caption(
                    "The chart ranks the input signals used by the AI model. "
                    "Longer bars indicate greater average influence on predictions. "
                    "These are SHAP importance values, not percentages."
                )

                import altair as alt

                chart_data = view[
                    ["friendly_name", "mean_abs_shap"]
                ].copy()

                chart = (
                    alt.Chart(chart_data)
                    .mark_bar(color="#1673D1", cornerRadiusEnd=4)
                    .encode(
                        x=alt.X(
                            "mean_abs_shap:Q",
                            title="Average absolute SHAP contribution"
                        ),
                        y=alt.Y(
                            "friendly_name:N",
                            sort="-x",
                            title=None,
                            axis=alt.Axis(labelLimit=260)
                        ),
                        tooltip=[
                            alt.Tooltip("friendly_name:N", title="Input signal"),
                            alt.Tooltip(
                                "mean_abs_shap:Q",
                                title="Mean absolute SHAP",
                                format=",.2f"
                            )
                        ]
                    )
                    .properties(height=350)
                )

                st.altair_chart(chart, use_container_width=True)

                st.markdown("#### What does the SHAP value mean?")
                st.info(
                    "**Mean absolute SHAP value** is the average size of a signal's "
                    "contribution to the model's predictions relative to a reference "
                    "prediction, regardless of whether it pushes a prediction up or down. "
                    "Longer bars mean greater average influence across the evaluated samples."
                )
                st.caption(
                    "The values are not percentages, forecasting errors, or traffic congestion "
                    "measurements. They do not show whether a signal raises or lowers an "
                    "individual prediction. Correlated signals can share predictive information."
                )

                st.markdown("#### Which signals influence the model most?")
                top_two = view.nlargest(2, "mean_abs_shap")
                if len(top_two) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Strongest signal", top_two.iloc[0]["friendly_name"])
                    with col2:
                        st.metric("Second strongest signal", top_two.iloc[1]["friendly_name"])
                    st.write(
                        f"For **{horizon_label(h)}**, **{top_two.iloc[0]['friendly_name']}** "
                        f"has the largest average influence, followed by "
                        f"**{top_two.iloc[1]['friendly_name']}**. Other signals have "
                        "smaller individual contributions but may still be useful together."
                    )
                else:
                    st.write(
                        f"The most influential signal for {horizon_label(h)} is "
                        f"**{top_two.iloc[0]['friendly_name']}**."
                    )

                with st.expander("What does each input signal mean?", expanded=True):
                    explanations = {
                        "Latest observed traffic volume":
                            "The most recent traffic-volume measurement available at the forecast starting time (forecast origin).",
                        "Hour of day":
                            "The hour-of-day input used to recognize recurring daily traffic patterns.",
                        "Day of week":
                            "The day-of-week input used to recognize recurring weekly patterns.",
                        "Month of year":
                            "The calendar month, which may capture seasonal patterns.",
                        "Weekend":
                            "Indicates whether the time falls on a weekend.",
                        "Recent 3-hour average":
                            "Average traffic volume across a recent three-hour window.",
                        "Recent 24-hour average":
                            "Average traffic volume across a recent 24-hour window.",
                        "Recent 7-day average":
                            "Average traffic volume across a recent seven-day window."
                    }
                    for _, feature_row in view.iterrows():
                        signal = feature_row["friendly_name"]
                        original = str(feature_row["feature"]).replace("num__", "").replace("cat__", "")
                        if original.startswith("lag_"):
                            try:
                                hours = int(original.split("_")[-1])
                                meaning = f"Traffic volume recorded {hours} hour(s) before the forecast starting time."
                            except ValueError:
                                meaning = "A historical traffic measurement used by the model."
                        else:
                            meaning = explanations.get(
                                signal, "A historical or time-related input used by the forecasting model."
                            )
                        st.markdown(f"**{signal}** — {meaning}")
                    st.caption(
                        "Forecast origin means the time from which a prediction is made. "
                        "For example, a 1-hour-ahead forecast made at 8 AM predicts traffic "
                        "for 9 AM. Exact rolling-window alignment follows the research feature-engineering setup."
                    )

                with st.expander(
                    "🔬 Advanced details: View AI feature importance data", expanded=False
                ):
                    technical_data = view[
                        ["friendly_name", "feature", "mean_abs_shap"]
                    ].copy()
                    technical_data.columns = [
                        "Input signal", "Original feature name", "Mean absolute SHAP value"
                    ]
                    st.dataframe(
                        technical_data.round(2), hide_index=True, use_container_width=True
                    )
                    st.caption(
                        "Original feature names and saved global SHAP importance scores "
                        "are provided for research transparency."
                    )
            else:st.warning('The selected SHAP CSV has no valid importance values.')
        else:
            st.warning(f'The h{h} SHAP file is present but lacks feature and mean_abs_shap columns.')
    else:
        st.warning(f'SHAP results for {horizon_label(h)} could not be loaded. Verify research/shap_importance_h{h}.csv exists on the GitHub main branch.')
        with st.expander('Troubleshooting: expected paths'):
            st.code('\n'.join(str(p.relative_to(BASE)) for p in [BASE/f'shap_importance_h{h}.csv',BASE/'metrics'/f'shap_importance_h{h}.csv',BASE/'research'/f'shap_importance_h{h}.csv']))

elif page=='📊 Research Evidence':
    st.title('📊 Research Evidence')
    st.write('How accurately do different methods forecast future traffic volume? All results are from held-out historical observations, not live traffic.')
    comp = normalized_comparison()
    if comp is not None and not comp.empty:
        import altair as alt

        horizons = sorted(set(int(x) for x in comp.horizon_steps.dropna()))
        selected = st.selectbox('Choose forecast horizon', horizons, format_func=horizon_label)
        part = comp[comp.horizon_steps == selected].copy().sort_values('mae')
        if part.empty:
            st.warning('No model comparisons are available for this horizon.')
        else:
            best = part.iloc[0]
            rf = metric_row(selected, 'rf')
            lgbm = metric_row(selected, 'lgbm')
            profile = metric_row(selected, 'historical_profile')
            st.subheader('Research performance at a glance')
            c1, c2, c3 = st.columns(3)
            c1.metric('AI model used in this demo', 'LightGBM',
                      help='The saved predictions and uncertainty displays in the interactive demo use LightGBM.')
            c2.metric('LightGBM average forecast error',
                      f'{float(lgbm.mae):,.1f} vehicles/hour' if lgbm is not None else 'Not available',
                      help='Mean absolute error (MAE) for the selected horizon on the held-out historical test set. Lower is better.')
            if lgbm is not None and profile is not None and float(profile.mae) > 0:
                lgbm_gain = 100 * (float(profile.mae) - float(lgbm.mae)) / float(profile.mae)
                gain_text = f'{lgbm_gain:.1f}%'
            else:
                gain_text = 'Not available'
            c3.metric('LightGBM error reduction vs historical profile', gain_text,
                      help='Relative reduction in historical prediction MAE versus the simple baseline, not reduced congestion.')
            if lgbm is not None and rf is not None:
                if float(rf.mae) < float(lgbm.mae):
                    st.info(
                        f'**Why is LightGBM highlighted?** It powers the interactive demonstration '
                        f'and has saved chronological validation results. Random Forest achieved '
                        f'a lower overall test error for {horizon_label(selected)} '
                        f'(**{float(rf.mae):,.1f}** versus **{float(lgbm.mae):,.1f}** vehicles/hour). '
                        'The full comparison is shown below; these are different evaluation purposes.'
                    )
                else:
                    st.info('**Why is LightGBM highlighted?** It powers the saved interactive forecasts. '
                            'The complete held-out comparison, including Random Forest, is shown below.')
            else:
                st.caption('LightGBM powers the saved interactive forecasts. The full research comparison is shown below.')
            st.caption('All metrics describe historical forecasting errors, not measured improvements in road conditions.')

            st.subheader('Which forecasting method predicts traffic most accurately?')
            st.write('**Lower bars are better:** each bar shows average prediction error in vehicles per hour (MAE).')
            plot = part[['model', 'mae']].copy()
            plot['Method'] = plot['model'].map(model_name)
            ml_ids = {'rf', 'lgbm', 'ridge'}
            plot['Category'] = plot['model'].astype(str).str.lower().map(
                lambda x: 'Machine-learning model' if x in ml_ids else 'Reference baseline'
            )
            chart = (
                alt.Chart(plot)
                .mark_bar(cornerRadiusEnd=3)
                .encode(
                    x=alt.X('mae:Q', title='Mean absolute error (vehicles/hour)', scale=alt.Scale(zero=True)),
                    y=alt.Y('Method:N', sort=alt.EncodingSortField(field='mae', order='ascending'),
                            title=None, axis=alt.Axis(labelLimit=230)),
                    color=alt.Color('Category:N', title='Forecasting approach',
                                    scale=alt.Scale(domain=['Machine-learning model', 'Reference baseline'],
                                                    range=['#1874bb', '#8796a8'])),
                    tooltip=[alt.Tooltip('Method:N'), alt.Tooltip('Category:N'),
                             alt.Tooltip('mae:Q', title='Average error (vehicles/hour)', format=',.2f')]
                )
                .properties(height=max(280, len(plot) * 43))
            )
            st.altair_chart(chart, use_container_width=True)
            if rf is not None and lgbm is not None:
                summary = (f'**What the results show for {horizon_label(selected)}:** '
                           f'Random Forest has an average error of **{float(rf.mae):,.1f}** vehicles/hour; '
                           f'LightGBM has **{float(lgbm.mae):,.1f}** vehicles/hour and supplies the saved forecasts in this demonstration.')
                if profile is not None:
                    summary += f' The historical-profile baseline has **{float(profile.mae):,.1f}** vehicles/hour of error.'
                st.info(summary)
            else:
                st.info(f'The lowest-error method in this comparison is **{model_name(best["model"])}**. Lower MAE indicates more accurate predictions on the held-out dataset.')

            with st.expander('What does each forecasting method mean?', expanded=False):
                st.markdown('**Machine-learning models** — fitted to historical data:')
                st.markdown('**Random Forest:** Combines many decision trees to predict future traffic volume.\n\n'
                            '**LightGBM:** Uses boosted decision trees to learn traffic patterns; its saved predictions power the interactive demo.\n\n'
                            '**Ridge regression:** A regularized linear regression model relating historical inputs to future traffic.')
                st.markdown('**Reference baselines** — simpler forecasting approaches used for comparison:')
                st.markdown('**Current traffic (persistence):** Uses the latest observed traffic volume as the future forecast.\n\n'
                            '**Previous day:** Uses traffic recorded at the corresponding time on the previous day.\n\n'
                            '**Previous week:** Uses traffic recorded at the corresponding time one week earlier.\n\n'
                            '**Historical profile:** Uses typical traffic patterns learned from historical training data; exact grouping follows the experiment configuration.')
                st.caption('Baselines are legitimate forecasting methods, not necessarily trained AI models. They help test whether a more complex model provides added value.')

            st.subheader('How reliable are the predictions?')
            conf = optional_csv('conformal.csv')
            if conf is not None:
                conf.columns = conf.columns.str.strip().str.lower()
                if {'horizon_steps', 'model', 'coverage'}.issubset(conf.columns):
                    cc = conf[(pd.to_numeric(conf.horizon_steps, errors='coerce') == selected) &
                              (conf.model.astype(str).str.lower() == 'lgbm')]
                    if not cc.empty:
                        raw_coverage = float(pd.to_numeric(cc.iloc[0]['coverage'], errors='coerce'))
                        if np.isfinite(raw_coverage) and 0 <= raw_coverage <= 1:
                            coverage = raw_coverage * 100
                            st.metric('Observed prediction-interval coverage (LightGBM)', f'{coverage:.1f}%',
                                      help='Share of historical test observations within the saved prediction intervals; nominal target is 90%.')
                            st.write(f'**In simple terms:** About **{coverage:.0f} out of every 100** historical observations fell within the predicted uncertainty ranges. The target was 90 out of 100.')
                            st.caption('Coverage is not forecast accuracy. Intervals can miss observations and coverage may differ across time periods; future coverage is not guaranteed.')

            st.subheader('How does LightGBM forecast accuracy change across time periods?')
            rolling = optional_csv('rolling_origin.csv')
            if rolling is not None:
                rolling = rolling.rename(columns={c: c.strip().lower() for c in rolling.columns})
                if {'fold', 'horizon_steps', 'model', 'mae'}.issubset(rolling.columns):
                    rolling['model_key'] = rolling['model'].astype(str).str.strip().str.lower().replace({
                        'random forest': 'rf', 'random_forest': 'rf',
                        'lightgbm': 'lgbm', 'light_gbm': 'lgbm'
                    })
                    folds = rolling[
                        (pd.to_numeric(rolling['horizon_steps'], errors='coerce') == selected) &
                        (rolling['model_key'].isin(['rf', 'lgbm']))
                    ].copy()
                    folds['mae'] = pd.to_numeric(folds['mae'], errors='coerce')
                    folds['fold_number'] = pd.to_numeric(folds['fold'], errors='coerce')
                    folds = folds.dropna(subset=['mae', 'fold_number']).sort_values('fold_number')
                    if not folds.empty:
                        folds['Evaluation period'] = folds['fold_number'].map(lambda x: f'Period {int(x)}')
                        folds['Method'] = folds['model_key'].map(model_name)
                        fold_order = [f'Period {int(x)}' for x in sorted(folds['fold_number'].unique())]
                        methods = sorted(folds['Method'].unique())
                        if len(methods) == 1:
                            st.caption(
                                f'Only **{methods[0]}** has saved rolling-origin results for '
                                f'**{horizon_label(selected)}**. The overall comparison above evaluates '
                                'other methods on a separate held-out test set; it does not imply that '
                                'they have rolling-origin results.'
                            )
                        else:
                            st.caption('Saved rolling-origin results compare ' + ' and '.join(methods) +
                                       ' across chronological evaluation periods.')
                        st.write('Each period is a separate chronological evaluation window. **Lower average error is better.**')
                        fold_chart = (
                            alt.Chart(folds)
                            .mark_bar(cornerRadiusEnd=3)
                            .encode(
                                x=alt.X('Evaluation period:N', sort=fold_order,
                                        title='Chronological evaluation period',
                                        axis=alt.Axis(labelAngle=0)),
                                xOffset=alt.XOffset('Method:N') if len(methods) > 1 else alt.value(0),
                                y=alt.Y('mae:Q', title='Average error (vehicles/hour)',
                                        scale=alt.Scale(zero=True)),
                                color=alt.Color('Method:N', title='Evaluated method',
                                                scale=alt.Scale(domain=['LightGBM', 'Random Forest'],
                                                                range=['#1874bb', '#0e6475'])),
                                tooltip=[alt.Tooltip('Evaluation period:N'), alt.Tooltip('Method:N'),
                                         alt.Tooltip('mae:Q', title='MAE (vehicles/hour)', format=',.2f')]
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(fold_chart, use_container_width=True)
                        if len(methods) == 1:
                            series = folds.groupby('fold_number', as_index=False)['mae'].mean().sort_values('fold_number')
                            if len(series) >= 2:
                                first_error, last_error = float(series.iloc[0]['mae']), float(series.iloc[-1]['mae'])
                                change = last_error - first_error
                                direction = ('increased' if change > 0 else 'decreased' if change < 0 else 'remained unchanged')
                                st.write(
                                    f'**What this shows:** {methods[0]} average error {direction} '
                                    f'from **{first_error:,.2f}** to **{last_error:,.2f} vehicles/hour** '
                                    f'between the first and last saved evaluation periods. '
                                    'Results can vary across time; this is not the overall test-set MAE.'
                                )
                        else:
                            st.write('**What this shows:** Compare the bar heights for each method across '
                                     'periods to see how its forecasting error varies over time. '
                                     'These are separate evaluations from the overall test comparison.')
                        with st.expander('View exact error for each evaluation period', expanded=False):
                            fold_table = folds[['Evaluation period', 'Method', 'mae']].rename(
                                columns={'mae': 'Average error (vehicles/hour)'})
                            st.dataframe(fold_table.round(2), hide_index=True, use_container_width=True)
                    else:
                        st.caption('No saved LightGBM or Random Forest rolling-origin results are available for this horizon.')
                else:
                    st.caption('Rolling-origin CSV is missing the fields needed to display this evaluation.')
            else:
                st.caption('Rolling-origin evaluation data are not available in the repository.')

            with st.expander('Advanced research details: Full comparison and download', expanded=False):
                display = part.copy()
                display['model'] = display['model'].map(model_name)
                display = display.rename(columns={
                    'horizon_steps': 'Forecast horizon (hours)',
                    'model': 'Forecasting method',
                    'n': 'Test examples',
                    'mae': 'Average absolute error (MAE)',
                    'rmse': 'Root mean squared error (RMSE)',
                    'r2': 'Explained variation (R²)'
                })
                st.dataframe(display.round(3), hide_index=True, use_container_width=True)
                st.caption('MAE is the average absolute prediction error; lower is better. RMSE gives more weight to large errors; lower is better. R² summarizes explained variation; higher is generally better.')
                st.download_button('Download original model comparison CSV',
                                   comp.to_csv(index=False).encode(), 'model_comparison.csv', 'text/csv')

            with st.expander('Advanced research validation (optional)', expanded=False):
                st.caption('These saved research tables are optional technical evidence, not additional live system capabilities.')
                for filename, heading in [
                    ('paired_bootstrap.csv', 'Paired model comparisons'),
                    ('missing_history_sensitivity.csv', 'Sensitivity to missing traffic history'),
                    ('monthly_coverage.csv', 'Prediction-interval coverage by month'),
                    ('subgroups.csv', 'Peak and off-peak performance')
                ]:
                    data = optional_csv(filename)
                    if data is None or data.empty:
                        continue
                    st.markdown(f'**{heading}**')
                    show = data.copy()
                    if 'horizon_steps' in show.columns:
                        show = show[pd.to_numeric(show['horizon_steps'], errors='coerce') == selected].copy()
                    if show.empty:
                        st.caption('No saved records for the selected forecast horizon.')
                        continue
                    if filename == 'paired_bootstrap.csv':
                        if 'comparison' in show.columns:
                            def readable_comparison(raw):
                                raw = str(raw)
                                if '_minus_' not in raw:
                                    return raw.replace('_', ' ').title()
                                left, right = raw.split('_minus_', 1)
                                return f'{model_name(left)} minus {model_name(right)}'
                            show['comparison'] = show['comparison'].map(readable_comparison)
                        show = show.rename(columns={
                            'horizon_steps': 'Forecast horizon (hours)',
                            'comparison': 'Compared methods',
                            'n_common': 'Shared test examples',
                            'delta_MAE': 'MAE difference',
                            'CI_low': 'Confidence interval lower',
                            'CI_high': 'Confidence interval upper',
                            'block_steps': 'Bootstrap block length'
                        })
                        st.caption('MAE difference = first method minus second method. Negative favors the first; positive favors the second. A confidence interval excluding zero supports a difference under the bootstrap assumptions. This table is optional technical evidence.')
                    st.dataframe(show.round(3), hide_index=True, use_container_width=True)

    else:
        st.error('Model comparison cannot be displayed. Verify model_comparison.csv exists and contains horizon_steps, model and mae columns.')
        with st.expander('Diagnostic: file locations', expanded=False):
            st.write('App folder:', str(BASE))
            st.write('Expected paths:', [str(p) for p in (BASE/'model_comparison.csv', BASE/'metrics/model_comparison.csv', BASE/'research/model_comparison.csv')])

    st.caption('**Study limitation:** Historical single-site traffic-volume forecasting does not establish measured congestion reduction, transferability to other cities, or effectiveness of traffic interventions.')

else:
    st.title('ℹ️ About & Roadmap')
    st.subheader('Research demonstration')
    st.write('UrbanFlow Sentinel AI is an explainable traffic-demand forecasting research prototype. It illustrates how historical traffic-volume observations can support short-term prediction and uncertainty-aware decision support.')
    st.subheader('How it works')
    a,b,c,d=st.columns(4)
    for col,emoji,title,detail in [(a,'📥','Historical data','Hourly traffic observations'),(b,'🧹','Quality checks','Missingness and temporal features'),(c,'🤖','AI forecast','Models estimate future volume'),(d,'📊','Interpret','Forecast, uncertainty and validation')]:
        with col:
            st.markdown(f'<div class="card"><h4>{emoji} {title}</h4><p>{detail}</p></div>',unsafe_allow_html=True)
    st.info('**Model distinction:** LightGBM supplies the saved interactive forecasts and chronological validation results. Random Forest achieved the lowest held-out MAE in the recorded model comparison; both are reported transparently in Research Evidence.')
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
