"""Rebuild histories for the first 100 test applicants and check the saved model.

Reads raw tables in chunks, retaining only selected applicants; this avoids
recomputing all applicants' feature tables just for an inference check.
"""
from pathlib import Path
import tempfile,argparse
import numpy as np
import pandas as pd
import joblib
from main import ROOT,features
from history import build_history
from extra_history import build_extra_history

def filter_table(source,destination,key,values):
    first=True
    for chunk in pd.read_csv(source,chunksize=500000):
        subset=chunk.loc[chunk[key].isin(values)]
        subset.to_csv(destination,mode='w' if first else 'a',header=first,index=False)
        first=False

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-cache',type=Path);args=parser.parse_args()
    test=pd.read_csv(ROOT/'data/application_test.csv',nrows=100)
    ids=set(test.SK_ID_CURR)
    with tempfile.TemporaryDirectory() as td:
        directory=Path(td)
        filter_table(ROOT/'data/bureau.csv',directory/'bureau.csv','SK_ID_CURR',ids)
        bureau=pd.read_csv(directory/'bureau.csv')
        filter_table(ROOT/'data/bureau_balance.csv',directory/'bureau_balance.csv','SK_ID_BUREAU',set(bureau.SK_ID_BUREAU))
        for name in ['previous_application','installments_payments','credit_card_balance','POS_CASH_balance']:
            filter_table(ROOT/'data'/f'{name}.csv',directory/f'{name}.csv','SK_ID_CURR',ids)
        history=build_history(directory).join(build_extra_history(directory),how='outer')
        x=features(test);x.index=test.SK_ID_CURR.values;x=x.join(history)
        if args.prepare_cache:
            joblib.dump({'features':x,'ids':test.SK_ID_CURR.values},args.prepare_cache)
            print('Independent inference features prepared:',x.shape,flush=True)
            return
        model=joblib.load(ROOT/'output/full_history/model.joblib')
        columns=model.named_steps['preprocess'].feature_names_in_
        for col in columns:
            if col not in x and (col.startswith('BUREAU_CREDIT_ACTIVE_') or col.startswith('BUREAU_CREDIT_TYPE_')):
                x[col]=np.where(x['BUREAU_COUNT'].notna(),0,np.nan)
            elif col not in x and col.startswith('BALANCE_STATUS_'):
                x[col]=np.where(x['BALANCE_MONTH_COUNT'].notna(),0,np.nan)
        x=x.reindex(columns=columns)
        numeric=x.select_dtypes(include='number').columns;x[numeric]=x[numeric].astype('float32')
        probabilities=model.predict_proba(x)[:,1]
        saved=pd.read_csv(ROOT/'output/full_history/submission.csv',nrows=100)
        assert np.array_equal(saved.SK_ID_CURR,test.SK_ID_CURR)
        assert np.allclose(probabilities,saved.TARGET,atol=1e-10,rtol=1e-10),float(np.max(np.abs(probabilities-saved.TARGET)))
        message='Saved complete-history model reproduces first 100 predictions from independently rebuilt history features.'
        (ROOT/'output/inference_verification.txt').write_text(message+'\n');print(message)
if __name__=='__main__':main()
