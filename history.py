"""Applicant-level history features. Balance CSV is processed in chunks."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

def build_history(data_dir):
    data_dir=Path(data_dir)
    bureau=pd.read_csv(data_dir/'bureau.csv')
    if bureau.SK_ID_BUREAU.duplicated().any():
        raise ValueError('Duplicate bureau IDs make balance joins ambiguous')
    cols=[c for c in bureau.select_dtypes(include='number') if c not in ['SK_ID_CURR','SK_ID_BUREAU']]
    agg=bureau.groupby('SK_ID_CURR')[cols].agg(['mean','max','min','sum'])
    agg.columns=[f'BUREAU_{a}_{b}' for a,b in agg.columns]
    agg['BUREAU_COUNT']=bureau.groupby('SK_ID_CURR').size()
    for col in ['CREDIT_ACTIVE','CREDIT_TYPE']:
        counts=pd.crosstab(bureau.SK_ID_CURR,bureau[col])
        counts.columns=[f'BUREAU_{col}_'+re.sub(r'[^A-Za-z0-9_]', '_', str(label)) for label in counts.columns]
        agg=agg.join(counts)
    mapping=bureau.set_index('SK_ID_BUREAU').SK_ID_CURR
    totals=None;month_min=None;month_max=None;rows=0;unmatched=0
    for chunk in pd.read_csv(data_dir/'bureau_balance.csv',chunksize=500000,dtype={'STATUS':'string'}):
        rows+=len(chunk);chunk['SK_ID_CURR']=chunk.SK_ID_BUREAU.map(mapping)
        unmatched+=int(chunk.SK_ID_CURR.isna().sum());chunk=chunk.dropna(subset=['SK_ID_CURR'])
        counts=pd.crosstab(chunk.SK_ID_CURR,chunk.STATUS).add_prefix('BALANCE_STATUS_')
        counts['BALANCE_MONTH_COUNT']=chunk.groupby('SK_ID_CURR').size()
        totals=counts if totals is None else totals.add(counts,fill_value=0)
        lo=chunk.groupby('SK_ID_CURR').MONTHS_BALANCE.min();hi=chunk.groupby('SK_ID_CURR').MONTHS_BALANCE.max()
        month_min=lo if month_min is None else pd.concat([month_min,lo],axis=1).min(axis=1)
        month_max=hi if month_max is None else pd.concat([month_max,hi],axis=1).max(axis=1)
    if totals is not None:
        totals=totals.fillna(0)
        totals['BALANCE_MONTH_MIN']=month_min;totals['BALANCE_MONTH_MAX']=month_max
        late=[c for c in totals if c in [f'BALANCE_STATUS_{i}' for i in range(1,6)]]
        totals['BALANCE_DELINQUENT_FRACTION']=totals[late].sum(axis=1)/totals.BALANCE_MONTH_COUNT.replace(0,np.nan)
        agg=agg.join(totals)
    agg.attrs['bureau_rows']=len(bureau);agg.attrs['balance_rows']=rows;agg.attrs['unmatched_balance_rows']=unmatched
    print(f'History ready: bureau={len(bureau):,}; balance={rows:,}; unmatched={unmatched:,}; features={len(agg.columns)}',flush=True)
    return agg.astype('float32')
