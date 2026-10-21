"""Home Credit binary classification. Demo outputs are NOT competition results."""
from pathlib import Path
import argparse, json, warnings, platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
import joblib
from lightgbm import LGBMClassifier

SEED = 42
ROOT = Path(__file__).resolve().parent

def demo_data(n=5000):
    rng = np.random.default_rng(SEED)
    x = pd.DataFrame({'SK_ID_CURR':np.arange(100000,100000+n), 'AMT_INCOME_TOTAL':rng.lognormal(11.7,.5,n), 'AMT_CREDIT':rng.lognormal(13,.6,n), 'AMT_ANNUITY':rng.lognormal(10,.5,n), 'DAYS_BIRTH':-rng.integers(7500,25000,n), 'DAYS_EMPLOYED':-rng.integers(0,14000,n), 'EXT_SOURCE_1':rng.uniform(0,1,n), 'EXT_SOURCE_2':rng.uniform(0,1,n), 'EXT_SOURCE_3':rng.uniform(0,1,n), 'NAME_EDUCATION_TYPE':rng.choice(['Secondary','Higher','Incomplete'],n), 'NAME_INCOME_TYPE':rng.choice(['Working','Pensioner','Commercial'],n), 'CNT_CHILDREN':rng.poisson(.5,n)})
    z = -3.5 + 3*(.5-x.EXT_SOURCE_2) + 2*(.5-x.EXT_SOURCE_3) + .35*np.log(x.AMT_CREDIT/x.AMT_INCOME_TOTAL) + .55*(x.NAME_EDUCATION_TYPE=='Secondary')
    y = rng.binomial(1,1/(1+np.exp(-z)))
    for col in ['EXT_SOURCE_1','EXT_SOURCE_3','AMT_ANNUITY']:
        x.loc[rng.random(n)<.15,col]=np.nan
    x.loc[rng.random(n)<.03,'DAYS_EMPLOYED']=365243
    x['TARGET']=y
    return x.iloc[:4000].copy(),x.iloc[4000:].drop(columns='TARGET').copy()

def features(frame):
    x=frame.drop(columns=['TARGET','SK_ID_CURR'],errors='ignore').copy()
    if 'DAYS_EMPLOYED' in x:
        x['EMPLOYMENT_SENTINEL']=(x.DAYS_EMPLOYED==365243).astype(int)
        x['DAYS_EMPLOYED']=x.DAYS_EMPLOYED.replace(365243,np.nan)
    for new,a,b in [('CREDIT_INCOME_RATIO','AMT_CREDIT','AMT_INCOME_TOTAL'),('ANNUITY_INCOME_RATIO','AMT_ANNUITY','AMT_INCOME_TOTAL'),('ANNUITY_CREDIT_RATIO','AMT_ANNUITY','AMT_CREDIT'),('EMPLOYED_BIRTH_RATIO','DAYS_EMPLOYED','DAYS_BIRTH')]:
        if a in x and b in x: x[new]=x[a]/x[b].replace(0,np.nan)
    ext=[c for c in ['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3'] if c in x]
    if ext: x['EXT_SOURCE_MEAN']=x[ext].mean(axis=1)
    return x.replace([np.inf,-np.inf],np.nan)

def secondary_tables(x, data_dir):
    """Aggregate historical numeric records to applicant level; never use TARGET."""
    for name in ['bureau','previous_application','installments_payments','credit_card_balance','POS_CASH_balance']:
        path=data_dir/f'{name}.csv'
        if not path.exists(): continue
        df=pd.read_csv(path)
        if 'SK_ID_CURR' not in df: raise ValueError(f'{path}: missing SK_ID_CURR')
        cols=[c for c in df.select_dtypes(include='number') if c not in ['SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','TARGET']]
        agg=df.groupby('SK_ID_CURR')[cols].agg(['mean','max','min'])
        agg.columns=[f'{name.upper()}_{a}_{b}' for a,b in agg.columns]
        agg[f'{name.upper()}_COUNT']=df.groupby('SK_ID_CURR').size()
        x=x.join(agg)
    # bureau_balance is linked through SK_ID_BUREAU, not directly to applicants.
    bp=data_dir/'bureau_balance.csv'; b=data_dir/'bureau.csv'
    if bp.exists() and b.exists():
        bb=pd.read_csv(bp); bureau=pd.read_csv(b,usecols=['SK_ID_BUREAU','SK_ID_CURR'])
        linked=bb.merge(bureau,on='SK_ID_BUREAU',validate='many_to_one')
        counts=pd.crosstab(linked.SK_ID_CURR,linked.STATUS).add_prefix('BUREAU_BALANCE_STATUS_')
        x=x.join(counts)
    return x

def pipeline(x, model, compact=False):
    cats=x.select_dtypes(include=['object','category','string']).columns.tolist()
    nums=[c for c in x if c not in cats]
    pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]),nums),('cat',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Missing',keep_empty_features=True)),('encode',OneHotEncoder(handle_unknown='ignore',min_frequency=20,sparse_output=True,dtype=np.float32 if compact else np.float64))]),cats)])
    return Pipeline([('preprocess',pre),('model',model)])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--demo',action='store_true'); parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--output-dir',type=Path); parser.add_argument('--folds',type=int,default=3)
    parser.add_argument('--secondary',action='store_true'); parser.add_argument('--sample',type=int)
    args=parser.parse_args()
    if args.folds<2: parser.error('--folds must be at least 2')
    out=args.output_dir or ROOT/'output'/('demo' if args.demo else 'real');out.mkdir(parents=True,exist_ok=True)
    if args.demo:
        train,test=demo_data()
        if args.secondary: parser.error('--secondary needs real CSV files')
    else:
        required=[args.data_dir/'application_train.csv',args.data_dir/'application_test.csv']
        if not all(p.exists() for p in required): parser.error('Download both application CSVs from Kaggle, or explicitly use --demo.')
        train,test=[pd.read_csv(p) for p in required]
    if args.sample: train=train.sample(min(args.sample,len(train)),random_state=SEED)
    for frame in [train,test]:
        if frame.SK_ID_CURR.duplicated().any(): raise ValueError('Applicant IDs must be unique')
    y=train.TARGET.astype(int)
    if set(y.unique())!={0,1}: raise ValueError('TARGET must contain binary 0 and 1')
    x=features(train); xt=features(test).reindex(columns=x.columns)
    x.index=train.SK_ID_CURR.values;xt.index=test.SK_ID_CURR.values;y.index=x.index
    if args.secondary:
        x=secondary_tables(x,args.data_dir);xt=secondary_tables(xt,args.data_dir).reindex(columns=x.columns)
    missing=train.isna().mean().sort_values(ascending=False).rename('missing_fraction');missing.to_csv(out/'missing_values.csv')
    fig,ax=plt.subplots();missing.head(20).sort_values().plot.barh(ax=ax);ax.set_title('Missing fractions');fig.tight_layout();fig.savefig(out/'missing_values.png');plt.close(fig)
    fig,ax=plt.subplots();y.value_counts().sort_index().plot.bar(ax=ax);ax.set_title('Target distribution (1 = default)');fig.tight_layout();fig.savefig(out/'target_distribution.png');plt.close(fig)
    cols=x.select_dtypes(include='number').columns[:12];corr=x[cols].corr()
    fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(len(cols)),cols,rotation=90);ax.set_yticks(range(len(cols)),cols);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/'correlation.png');plt.close(fig)
    # This holdout is never used for model/parameter/threshold selection.
    xd,xh,yd,yh=train_test_split(x,y,test_size=.2,stratify=y,random_state=SEED)
    cv=StratifiedKFold(args.folds,shuffle=True,random_state=SEED)
    print(f'Data ready: {len(train):,} training rows, {len(test):,} test rows; {args.folds}-fold CV.',flush=True)
    models={'Logistic regression':LogisticRegression(max_iter=1500,random_state=SEED),'Random forest':RandomForestClassifier(n_estimators=120,max_depth=10,min_samples_leaf=20,n_jobs=2,random_state=SEED),'LightGBM':LGBMClassifier(n_estimators=180,learning_rate=.05,num_leaves=15,min_child_samples=40,reg_lambda=1,verbosity=-1,n_jobs=2,random_state=SEED)}
    rows=[];fitted={};oofs={}
    for name,model in models.items():
        print(f'Training {name}...',flush=True)
        p=pipeline(xd,model)
        grid={'model__C':[.1,1]} if name=='Logistic regression' else ({'model__max_depth':[6,10]} if name=='Random forest' else {'model__num_leaves':[7,15],'model__reg_lambda':[1,5]})
        search=GridSearchCV(p,grid,scoring='roc_auc',cv=cv,n_jobs=1,refit=True)
        search.fit(xd,yd);best=search.best_estimator_; fitted[name]=best
        # OOF of the selected configuration; selection makes this estimate optimistic.
        from sklearn.base import clone
        oof=np.zeros(len(yd));folds=[]
        for a,b in cv.split(xd,yd):
            fold=clone(best);fold.fit(xd.iloc[a],yd.iloc[a]);oof[b]=fold.predict_proba(xd.iloc[b])[:,1];folds.append(roc_auc_score(yd.iloc[b],oof[b]))
        oofs[name]=oof
        pd.DataFrame({'SK_ID_CURR':xd.index,'actual':yd.values,'probability':oof}).to_csv(out/(name.lower().replace(' ','_')+'_oof.csv'),index=False)
        search.cv_results_['params']=[json.dumps(v) for v in search.cv_results_['params']]
        pd.DataFrame(search.cv_results_).to_csv(out/(name.lower().replace(' ','_')+'_search.csv'),index=False)
        rows.append({'model':name,'selection_cv_auc':search.best_score_,'fold_auc_std':np.std(folds),'oof_auc':roc_auc_score(yd,oof),'parameters':json.dumps(search.best_params_)})
        print(name,rows[-1],flush=True)
    table=pd.DataFrame(rows).sort_values('selection_cv_auc',ascending=False)
    winner=table.iloc[0]['model'];best=fitted[winner]
    prec,rec,thresholds=precision_recall_curve(yd,oofs[winner]); f1=2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-12)
    threshold=float(thresholds[np.argmax(f1)])
    probabilities={name:p.predict_proba(xh)[:,1] for name,p in fitted.items()}
    table['holdout_auc']=table.model.map({n:roc_auc_score(yh,p) for n,p in probabilities.items()})
    table['holdout_average_precision']=table.model.map({n:average_precision_score(yh,p) for n,p in probabilities.items()});table.to_csv(out/'model_comparison.csv',index=False)
    p=probabilities[winner];pred=(p>=threshold).astype(int)
    metrics={'data_mode':'SYNTHETIC_DEMO' if args.demo else 'HOME_CREDIT','selected_model':winner,'selection_rule':'highest development CV AUC, before holdout evaluation','threshold_from_development_oof':threshold,'holdout_auc':roc_auc_score(yh,p),'holdout_average_precision':average_precision_score(yh,p),'precision':precision_score(yh,pred,zero_division=0),'recall':recall_score(yh,pred,zero_division=0),'f1':f1_score(yh,pred,zero_division=0),'default_fraction':float(y.mean()),'rows':len(train),'features':len(x.columns),'python':platform.python_version(),'seed':SEED,'cv_note':'CV and OOF are used for tuning; final holdout is the independent evaluation. Threshold OOF is not nested.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    for kind in ['roc','pr']:
        fig,ax=plt.subplots()
        for name,prob in probabilities.items():
            if kind=='roc': a,b,_=roc_curve(yh,prob)
            else: b,a,_=precision_recall_curve(yh,prob)
            ax.plot(a,b,label=name)
        if kind=='roc': ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='False positive rate',ylabel='True positive rate')
        else: ax.axhline(yh.mean(),ls='--',color='gray');ax.set(xlabel='Recall',ylabel='Precision')
        ax.legend();ax.set_title(f'Holdout {kind.upper()}');fig.tight_layout();fig.savefig(out/f'{kind}_curve.png');plt.close(fig)
    fig,ax=plt.subplots();ConfusionMatrixDisplay(confusion_matrix(yh,pred)).plot(ax=ax,colorbar=False,values_format='d');ax.set_title(f'{winner}, threshold={threshold:.3f}');fig.tight_layout();fig.savefig(out/'confusion_matrix.png');plt.close(fig)
    pd.DataFrame({'SK_ID_CURR':xh.index,'actual':yh.values,'probability':p,'predicted':pred,'error':pred!=yh.values}).to_csv(out/'holdout_predictions.csv',index=False)
    # Raw-feature permutation importance uses development data, not final holdout.
    xs=xd.sample(min(1500,len(xd)),random_state=SEED);ys=yd.loc[xs.index]
    imp=permutation_importance(best,xs,ys,scoring='roc_auc',n_repeats=3,random_state=SEED,n_jobs=1)
    importance=pd.Series(imp.importances_mean,index=x.columns).sort_values(ascending=False);importance.to_csv(out/'feature_importance.csv')
    fig,ax=plt.subplots(figsize=(8,6));importance.head(20).sort_values().plot.barh(ax=ax);ax.set_title('Development permutation importance (AUC drop)');fig.tight_layout();fig.savefig(out/'feature_importance.png');plt.close(fig)
    import shap
    tree=fitted['LightGBM']; transformed=tree.named_steps['preprocess'].transform(xs.iloc[:400])
    if hasattr(transformed,'toarray'): transformed=transformed.toarray()
    explanation=shap.TreeExplainer(tree.named_steps['model'])(transformed)
    names=tree.named_steps['preprocess'].get_feature_names_out()
    shap.summary_plot(explanation.values,transformed,feature_names=names,show=False,max_display=15)
    plt.title('LightGBM SHAP - development sample');plt.tight_layout();plt.savefig(out/'shap_summary.png',bbox_inches='tight');plt.close()
    from sklearn.base import clone
    final=clone(best);final.fit(x,y);testp=final.predict_proba(xt)[:,1]
    filename='demo_predictions_NOT_FOR_KAGGLE.csv' if args.demo else 'submission.csv'
    submission=pd.DataFrame({'SK_ID_CURR':test.SK_ID_CURR.values,'TARGET':testp})
    assert len(submission)==len(test) and submission.TARGET.between(0,1).all() and not submission.isna().any().any()
    submission.to_csv(out/filename,index=False);joblib.dump(final,out/'model.joblib')
    (out/'report.md').write_text('# Home Credit results\n\n'+('**Synthetic demo only; these are not real Home Credit scores.**\n\n' if args.demo else 'Real application CSV run.\n\n')+table.to_markdown(index=False)+'\n\n```json\n'+json.dumps(metrics,indent=2)+'\n```\n\nSHAP explains the LightGBM model; permutation importance explains the selected model. Neither establishes causation. No labels exist for Kaggle application_test, so no test AUC is reported.\n')
    print(json.dumps(metrics,indent=2),flush=True)

if __name__=='__main__': main()
