"""Compare a history-enriched LightGBM against the completed application baseline."""
from pathlib import Path
import argparse,json,shutil,gc
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.base import clone
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,roc_curve,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay
from lightgbm import LGBMClassifier
from main import features,pipeline,SEED,ROOT
from history import build_history
from extra_history import build_extra_history

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--history-cache',type=Path);parser.add_argument('--resume',action='store_true');parser.add_argument('--extra',action='store_true');args=parser.parse_args()
    base=ROOT/'output/real';out=ROOT/'output'/('full_history' if args.extra else 'enhanced');out.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(ROOT/'data/application_train.csv');test=pd.read_csv(ROOT/'data/application_test.csv')
    history=joblib.load(args.history_cache) if args.history_cache else build_history(ROOT/'data')
    history_meta=history.attrs.copy()
    model_name='LightGBM + bureau history'
    if args.extra:
        extra=build_extra_history(ROOT/'data')
        if not extra.attrs['row_counts']: parser.error('--extra requires at least one optional history CSV')
        history_meta['extra_tables']=extra.attrs['row_counts']
        history=history.join(extra.astype('float32'),how='outer')
        del extra
        gc.collect()
        model_name='LightGBM + all available history'
    x=features(train);x.index=train.SK_ID_CURR.values;x=x.join(history)
    xt=features(test);xt.index=test.SK_ID_CURR.values;xt=xt.join(history).reindex(columns=x.columns)
    y=pd.Series(train.TARGET.values,index=x.index)
    test_ids=test.SK_ID_CURR.values.copy()
    if args.extra:
        numeric=x.select_dtypes(include='number').columns
        x[numeric]=x[numeric].astype('float32');xt[numeric]=xt[numeric].astype('float32')
    del history,train,test
    gc.collect()
    xd,xh,yd,yh=train_test_split(x,y,test_size=.2,stratify=y,random_state=SEED)
    cv=StratifiedKFold(3,shuffle=True,random_state=SEED)
    model=LGBMClassifier(n_estimators=180,learning_rate=.05,num_leaves=15,min_child_samples=40,reg_lambda=1,verbosity=-1,n_jobs=2,random_state=SEED)
    search=GridSearchCV(pipeline(xd,model,compact=args.extra),{'model__num_leaves':[7,15],'model__reg_lambda':[1,5]},scoring='roc_auc',cv=cv,refit=True)
    print(f'History source rows: {history_meta}',flush=True)
    print(f'Training history LightGBM: {len(x):,} rows, {len(x.columns)} features',flush=True)
    checkpoint=out/'checkpoint.joblib'
    if args.resume and checkpoint.exists():
        saved=joblib.load(checkpoint)
        if saved.get('feature_columns') != x.columns.tolist(): raise ValueError('Checkpoint feature schema differs; rerun without --resume')
        best=saved['best'];oof=saved['oof'];folds=saved['folds'];cv_score=saved['cv_score'];best_params=saved['best_params']
    else:
        search.fit(xd,yd);best=search.best_estimator_
        pd.DataFrame(search.cv_results_).to_csv(out/'search.csv',index=False)
        oof=np.zeros(len(yd));folds=[]
        for i,(a,b) in enumerate(cv.split(xd,yd)):
            p=clone(best);p.fit(xd.iloc[a],yd.iloc[a]);oof[b]=p.predict_proba(xd.iloc[b])[:,1];folds.append(roc_auc_score(yd.iloc[b],oof[b]));print('History fold',i+1,folds[-1],flush=True)
        pd.DataFrame({'SK_ID_CURR':xd.index,'actual':yd.values,'probability':oof}).to_csv(out/'oof.csv',index=False)
        cv_score=float(search.best_score_);best_params=search.best_params_
        joblib.dump({'best':best,'oof':oof,'folds':folds,'cv_score':cv_score,'best_params':best_params,'feature_columns':x.columns.tolist()},checkpoint)
    if not (base/'holdout_predictions.csv').exists():
        print('History training checkpoint saved; rerun after baseline completes.',flush=True)
        return
    precision,recall,thresholds=precision_recall_curve(yd,oof);f1=2*precision[:-1]*recall[:-1]/(precision[:-1]+recall[:-1]+1e-12);threshold=float(thresholds[np.argmax(f1)])
    # Choose which configuration to deliver ONLY by development CV scores.
    comparison=pd.read_csv(base/'model_comparison.csv')
    if args.extra and (ROOT/'output/enhanced/metrics.json').exists():
        bureau_metrics=json.loads((ROOT/'output/enhanced/metrics.json').read_text())
        comparison=pd.concat([comparison,pd.DataFrame([{k:bureau_metrics[k] for k in comparison.columns}])],ignore_index=True)
    new={'model':model_name,'selection_cv_auc':cv_score,'fold_auc_std':np.std(folds),'oof_auc':roc_auc_score(yd,oof),'parameters':json.dumps(best_params)}
    baseline_max=float(comparison.selection_cv_auc.max());use_history=bool(cv_score>baseline_max)
    probability=best.predict_proba(xh)[:,1];pred=(probability>=threshold).astype(int)
    new['holdout_auc']=roc_auc_score(yh,probability);new['holdout_average_precision']=average_precision_score(yh,probability)
    comparison=pd.concat([comparison,pd.DataFrame([new])],ignore_index=True).sort_values('selection_cv_auc',ascending=False)
    comparison.to_csv(ROOT/'output/all_model_comparison.csv',index=False)
    metrics={**new,**history_meta,'data_mode':'HOME_CREDIT_REAL_HISTORY','training_rows':len(x),'test_rows':len(xt),'features':len(x.columns),'threshold_from_development_oof':threshold,'precision':precision_score(yh,pred,zero_division=0),'recall':recall_score(yh,pred),'f1':f1_score(yh,pred),'history_selected_by_cv':use_history,'seed':SEED,'folds':3,'cv_note':'Same 80/20 split as baseline; tuned CV/OOF are optimistic. Holdout was not used for selection.'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    hp=pd.DataFrame({'SK_ID_CURR':xh.index,'actual':yh.values,'probability':probability,'predicted':pred,'error':pred!=yh.values});hp.to_csv(out/'holdout_predictions.csv',index=False)
    baseline=pd.read_csv(base/'holdout_predictions.csv').set_index('SK_ID_CURR').loc[xh.index]
    assert np.array_equal(baseline.actual.values,yh.values)
    for curve in ['roc','pr']:
        fig,ax=plt.subplots()
        for label,prob in [('Main-table selected model',baseline.probability.values),(model_name,probability)]:
            if curve=='roc':a,b,_=roc_curve(yh,prob)
            else:b,a,_=precision_recall_curve(yh,prob)
            ax.plot(a,b,label=label)
        if curve=='roc':ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='False positive rate',ylabel='True positive rate')
        else:ax.axhline(yh.mean(),ls='--',color='gray');ax.set(xlabel='Recall',ylabel='Precision')
        ax.set_title('Same independent holdout: '+curve.upper());ax.legend();fig.tight_layout();fig.savefig(out/f'{curve}_curve.png');plt.close(fig)
    fig,ax=plt.subplots();ConfusionMatrixDisplay(confusion_matrix(yh,pred)).plot(ax=ax,colorbar=False,values_format='d');ax.set_title(f'History model; threshold={threshold:.3f}');fig.tight_layout();fig.savefig(out/'confusion_matrix.png');plt.close(fig)
    import shap
    xs=xd.sample(400,random_state=SEED);trans=best.named_steps['preprocess'].transform(xs)
    if hasattr(trans,'toarray'):trans=trans.toarray()
    names=best.named_steps['preprocess'].get_feature_names_out();explanation=shap.TreeExplainer(best.named_steps['model'])(trans)
    imp=pd.Series(np.abs(explanation.values).mean(axis=0),index=names).sort_values(ascending=False);imp.to_csv(out/'shap_importance.csv')
    shap.summary_plot(explanation.values,trans,feature_names=names,show=False,max_display=20);plt.title('History LightGBM SHAP - development sample');plt.tight_layout();plt.savefig(out/'shap_summary.png',bbox_inches='tight');plt.close()
    fig,ax=plt.subplots(figsize=(9,7));imp.head(20).sort_values().plot.barh(ax=ax);ax.set_title('Mean absolute SHAP contribution (development sample)');fig.tight_layout();fig.savefig(out/'feature_importance.png');plt.close(fig)
    print('Refitting history model on all labeled rows...',flush=True)
    final=clone(best)
    if args.extra:
        final.set_params(model__force_col_wise=True,model__histogram_pool_size=128)
        metrics['final_memory_settings']={'force_col_wise':True,'histogram_pool_size_MB':128}
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    del xd,xh,xs,trans,explanation,baseline,best,search
    if 'p' in locals(): del p
    if 'saved' in locals(): del saved
    gc.collect()
    import ctypes
    try: ctypes.CDLL(None).malloc_trim(0)
    except AttributeError: pass
    final.fit(x,y);prob=final.predict_proba(xt)[:,1]
    pd.DataFrame({'SK_ID_CURR':test_ids,'TARGET':prob}).to_csv(out/'submission.csv',index=False);joblib.dump(final,out/'model.joblib')
    winner_name=comparison.iloc[0].model
    source={'LightGBM + bureau history':'enhanced','LightGBM + all available history':'full_history'}.get(winner_name,'real')
    selected=ROOT/'output'/source
    shutil.copyfile(selected/'submission.csv',ROOT/'output/submission.csv')
    (ROOT/'output/selected_model.json').write_text(json.dumps({'source':selected.name,'selection':'highest development CV AUC across all compared configurations','cv_auc':float(comparison.iloc[0].selection_cv_auc),'model':comparison.iloc[0].model},indent=2))
    print(json.dumps(metrics,indent=2),flush=True)
if __name__=='__main__':main()
