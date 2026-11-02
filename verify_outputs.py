"""Check saved metrics, split separation, labels, probabilities and submission order."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,precision_score,recall_score,f1_score
ROOT=Path(__file__).resolve().parent

def main():
    train=pd.read_csv(ROOT/'data/application_train.csv',usecols=['SK_ID_CURR','TARGET']).set_index('SK_ID_CURR')
    test=pd.read_csv(ROOT/'data/application_test.csv',usecols=['SK_ID_CURR'])
    checks=[]
    holdout_ids=None
    for phase in [name for name in ['real','enhanced','full_history'] if (ROOT/'output'/name/'metrics.json').exists()]:
        out=ROOT/'output'/phase;m=json.loads((out/'metrics.json').read_text());hp=pd.read_csv(out/'holdout_predictions.csv')
        assert hp.SK_ID_CURR.is_unique and len(hp)==61503
        assert np.array_equal(hp.actual.values,train.loc[hp.SK_ID_CURR,'TARGET'].values)
        assert hp.probability.between(0,1).all() and hp.probability.notna().all()
        threshold=m['threshold_from_development_oof'];prediction=(hp.probability>=threshold).astype(int)
        assert np.array_equal(prediction,hp.predicted)
        for name,value in [('holdout_auc',roc_auc_score(hp.actual,hp.probability)),('holdout_average_precision',average_precision_score(hp.actual,hp.probability)),('precision',precision_score(hp.actual,prediction)),('recall',recall_score(hp.actual,prediction)),('f1',f1_score(hp.actual,prediction))]:
            assert np.isclose(value,m[name]),(phase,name,value,m[name])
        if holdout_ids is None:holdout_ids=set(hp.SK_ID_CURR)
        else:assert holdout_ids==set(hp.SK_ID_CURR)
        files=list(out.glob('*_oof.csv')) if phase=='real' else [out/'oof.csv']
        for file in files:
            oof=pd.read_csv(file)
            assert oof.SK_ID_CURR.is_unique and len(oof)==246008
            assert not set(oof.SK_ID_CURR)&holdout_ids
            assert set(oof.SK_ID_CURR)|holdout_ids==set(train.index)
            assert np.array_equal(oof.actual.values,train.loc[oof.SK_ID_CURR,'TARGET'].values)
            assert oof.probability.between(0,1).all()
        submission=pd.read_csv(out/'submission.csv')
        assert np.array_equal(submission.SK_ID_CURR,test.SK_ID_CURR)
        assert len(submission)==48744 and submission.TARGET.between(0,1).all() and not submission.isna().any().any()
        checks.append(f'{phase}: labels, metrics, disjoint OOF/holdout and ordered 48,744-row submission verified')
    table=pd.read_csv(ROOT/'output/all_model_comparison.csv');selected=json.loads((ROOT/'output/selected_model.json').read_text())
    assert selected['model']==table.sort_values('selection_cv_auc',ascending=False).iloc[0].model
    chosen=pd.read_csv(ROOT/'output'/selected['source']/'submission.csv');final=pd.read_csv(ROOT/'output/submission.csv');pd.testing.assert_frame_equal(chosen,final)
    sample=pd.read_csv(ROOT/'data/sample_submission.csv')
    assert list(sample.columns)==list(final.columns) and sample.SK_ID_CURR.equals(final.SK_ID_CURR)
    checks.append('Final submission matches sample_submission column names and ID order')
    checks.append('Final submission corresponds to the model chosen by development CV AUC')
    text='\n'.join(checks);(ROOT/'output/verification.txt').write_text(text+'\n');print(text)
if __name__=='__main__':main()
