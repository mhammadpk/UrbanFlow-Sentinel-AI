import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import joblib
P=Path(__file__).parent
meta=json.loads((P/'metadata.json').read_text())
st.set_page_config(page_title='UrbanFlow Sentinel AI',layout='wide')
st.title('UrbanFlow Sentinel AI | Research Demonstration')
st.warning('Research-only historical forecasting. Not live traffic sensing, measured congestion forecasting, or validated intervention advice.')
st.caption(meta['note'])
h=st.selectbox('Forecast horizon (sampling steps)',meta['horizons_steps'])
@st.cache_resource
def load_model(h): return joblib.load(P/f'model_h{h}.joblib')
model=load_model(h)
tab1,tab2=st.tabs(['Held-out historical predictions','Feature-complete CSV inference'])
with tab1:
 d=pd.read_csv(P/f'historical_predictions_h{h}.csv',parse_dates=['timestamp'])
 d['lower']=d['lower'].clip(lower=0) if meta['target']=='traffic_volume' else d['lower']
 st.line_chart(d.set_index('timestamp')[['actual','predicted','lower','upper']].tail(500))
 st.dataframe(d.tail(25),use_container_width=True)
with tab2:
 st.info('Upload ONLY feature-complete rows produced by the research feature pipeline, not arbitrary raw observations.')
 st.download_button('Download example feature schema',(P/f'example_features_h{h}.csv').read_bytes(),file_name=f'example_features_h{h}.csv')
 f=st.file_uploader('Feature-complete CSV',type='csv')
 if f is not None:
  try:
   x=pd.read_csv(f)
   if len(x)==0 or len(x)>10000:raise ValueError('Supply between 1 and 10,000 rows.')
   missing=[c for c in meta['features'] if c not in x]
   if missing:raise ValueError('Missing columns: '+', '.join(missing))
   for col in meta['features']:
    if col=='sensor_id':
     if x[col].isna().any() or x[col].astype(str).str.strip().eq('').any():raise ValueError('Invalid sensor_id')
     x[col]=x[col].astype(str)
    else:
     x[col]=pd.to_numeric(x[col],errors='coerce')
     if np.isinf(x[col].to_numpy(dtype=float)).any():raise ValueError('Infinite values in '+col)
   if x['origin_value'].isna().any():raise ValueError('origin_value must be observed at prediction time.')
   if meta['target']=='traffic_volume' and (x['origin_value']<0).any():raise ValueError('Negative traffic volume is invalid.')
   if x['hour'].dropna().between(0,23).all() == False:raise ValueError('hour outside 0–23')
   if x['dow'].dropna().between(0,6).all() == False:raise ValueError('dow outside 0–6')
   if x['month'].dropna().between(1,12).all() == False:raise ValueError('month outside 1–12')
   pred=model.predict(x[meta['features']]);q=meta['conformal_q'][str(h)]
   lower=pred-q
   if meta['target']=='traffic_volume':lower=np.maximum(0,lower)
   out=pd.DataFrame({'prediction':pred,'lower':lower,'upper':pred+q})
   st.dataframe(out,use_container_width=True)
   st.download_button('Download predictions',out.to_csv(index=False),file_name='predictions.csv')
  except Exception as e:st.error(f'Invalid input: {e}')
