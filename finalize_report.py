"""Build a readable Chinese report from completed, verified real-data outputs."""
from pathlib import Path
import json
import pandas as pd
root=Path(__file__).resolve().parent
out=root/'output/real'
m=json.loads((out/'metrics.json').read_text());t=pd.read_csv(out/'model_comparison.csv');hp=pd.read_csv(out/'holdout_predictions.csv')
train=pd.read_csv(root/'data/application_train.csv')
joined=hp.merge(train,on='SK_ID_CURR',validate='one_to_one',suffixes=('','_source'))
joined['error_type']='correct'
joined.loc[(joined.actual==1)&(joined.predicted==0),'error_type']='false_negative'
joined.loc[(joined.actual==0)&(joined.predicted==1),'error_type']='false_positive'
# Describe error groups using holdout predictions only; no model changes follow this analysis.
summary=joined.groupby('error_type').agg(applicants=('SK_ID_CURR','size'),mean_probability=('probability','mean'),mean_income=('AMT_INCOME_TOTAL','mean'),mean_credit=('AMT_CREDIT','mean'),mean_ext_source_2=('EXT_SOURCE_2','mean'))
summary.to_csv(out/'error_group_summary.csv')
fn=int(((hp.actual==1)&(hp.predicted==0)).sum());fp=int(((hp.actual==0)&(hp.predicted==1)).sum())
imp=pd.read_csv(out/'feature_importance.csv',index_col=0).iloc[:,0].head(5)
report=f'''# Home Credit：真实数据实验报告

## 任务与数据
预测申请人的还款困难风险，TARGET=1 为还款困难标签（项目中简称违约风险）。
本次使用用户提供的真实 Home Credit 主表：训练集 {len(train):,} 行，测试集 48,744 行。
训练集违约比例 {m['default_fraction']:.2%}。本次未加载二级历史表，不能回答二级表是否提高效果。

## 验证设计
固定 seed=42，分层留出 20%（{len(hp):,} 人）作为最终 holdout。
剩余 80% 用于三折分层交叉验证和小网格参数搜索。
缺失填补、标准化、类别编码在每个训练折内部拟合。
按开发集 CV AUC 选择模型，再用开发集 OOF 概率选择最大 F1 阈值，最后评估 holdout。
CV 与 OOF 参与了参数和阈值选择，带有选择偏差；holdout 是独立验证。

## 模型比较
{t.to_markdown(index=False, floatfmt='.5f')}

选中模型：**{m['selected_model']}**。
主指标 holdout ROC-AUC = **{m['holdout_auc']:.5f}**；average precision = **{m['holdout_average_precision']:.5f}**。
AUC 是排序能力，不是预测正确的比例。AP 的无信息基准接近违约比例。

## 阈值和错误
开发集选定阈值 {m['threshold_from_development_oof']:.5f}，不是固定使用 0.5。
独立 holdout 上 precision={m['precision']:.3f}，recall={m['recall']:.3f}，F1={m['f1']:.3f}。
漏判 {fn:,} 人，误判 {fp:,} 人。
阈值最大化 F1，不代表最优业务成本；实际借贷决策应按损失与容量另行制定。

{summary.to_markdown(floatfmt='.4f')}

这些分组仅用于诊断，不用于再次调整模型。均值可能受极端值影响，不能证明群体因果关系。

## 特征解释
开发集抽样 permutation importance 前五名：
{imp.to_frame('mean_AUC_drop').to_markdown(floatfmt='.5f')}

重要性是打乱单个特征后 AUC 的下降。相关特征会分摊重要性，抽样结果存在噪声。
SHAP 图解释 LightGBM 的输出贡献；其横轴为原始模型输出贡献，不是直接的概率百分点。

## 测试集预测
用选定模型配置对全部 307,511 行有标签数据重训，生成 48,744 行 submission.csv。
测试集没有 TARGET 真值，因此**没有测试集 AUC，也没有 Kaggle 线上分数**。
本地 holdout AUC 不能当作 Kaggle 排行榜成绩。

## 限制与后续
本次仅做小范围调参，未采用嵌套交叉验证；随机 holdout 不能证明时间外推能力。
没有处理公平性、概率校准、生产漂移和业务决策审批。
下一步可在开发集加入二级表并比较 CV；不能反复利用同一 holdout 调参。
'''
(out/'RESULTS_ZH.md').write_text(report)
print(report)

if (root/'output/enhanced/metrics.json').exists():
    enhanced=root/'output'/('full_history' if (root/'output/full_history/metrics.json').exists() else 'enhanced');e=json.loads((enhanced/'metrics.json').read_text());all_models=pd.read_csv(root/'output/all_model_comparison.csv');selected=json.loads((root/'output/selected_model.json').read_text())
    h=pd.read_csv(enhanced/'holdout_predictions.csv');j=h.merge(train,on='SK_ID_CURR',validate='one_to_one')
    j['error_type']='correct';j.loc[(j.actual==1)&(j.predicted==0),'error_type']='false_negative';j.loc[(j.actual==0)&(j.predicted==1),'error_type']='false_positive'
    groups=j.groupby('error_type').agg(applicants=('SK_ID_CURR','size'),mean_probability=('probability','mean'),mean_income=('AMT_INCOME_TOTAL','mean'),mean_credit=('AMT_CREDIT','mean'),mean_ext_source_2=('EXT_SOURCE_2','mean'))
    groups.to_csv(enhanced/'error_group_summary.csv')
    main_lgb=all_models.loc[all_models.model=='LightGBM'].iloc[0]
    important=pd.read_csv(enhanced/'shap_importance.csv',index_col=0).head(10)
    important.columns=['mean_abs_SHAP']
    extra_note=('本次还使用了以下额外历史表：'+json.dumps(e['extra_tables'],ensure_ascii=False)) if e.get('extra_tables') else '本次没有使用 previous_application、installments_payments、credit_card_balance 或 POS_CASH_balance。'
    extra_rows_md='\n'.join(f'| {name} | {count:,} |' for name,count in e.get('extra_tables',{}).items())
    overview=f'''# Home Credit 贷款违约预测：完整真实数据项目

## 本次完成了什么
使用 application_train / application_test 主表，完成逻辑回归、随机森林、LightGBM 三模型比较；再加入 bureau 和 bureau_balance 做中间比较，最后整合全部成功上传的历史表训练完整历史版 LightGBM。
所有指标来自真实上传的数据，没有使用模拟样本，也没有抽取部分训练行来代替全量实验。

| 输入表 | 实际行数 |
|---|---:|
| application_train | 307,511 |
| application_test | 48,744 |
| bureau | {e['bureau_rows']:,} |
| bureau_balance | {e['balance_rows']:,} |
{extra_rows_md}

十份文件中，八份数据表用于训练或预测；另两份是字段说明与提交格式示例，不作为预测特征。

训练标签中约 8.07% 为还款困难；最终预留 holdout 为 61,503 行。
历史表生成 {e['features']-126} 个特征，增强版共有 {e['features']} 个原始/工程特征（编码和缺失指示前）。
有 {e['unmatched_balance_rows']:,} 条 bureau_balance 记录的 SK_ID_BUREAU 无法在提供的 bureau 中关联，已排除，不猜测归属。
历史余额按块读取，不把 2,730 万条月度记录直接合并到主表，以避免重复申请人和内存膨胀。

## 模型表现
{all_models[['model','selection_cv_auc','fold_auc_std','holdout_auc','holdout_average_precision']].to_markdown(index=False,floatfmt='.5f')}

按开发集 CV AUC 选择 **{selected['model']}**，最终提交预测文件是 `output/submission.csv`。
加入历史表后，相比主表 LightGBM，CV AUC 变化为 {e['selection_cv_auc']-float(main_lgb.selection_cv_auc):+.5f}，独立 holdout AUC 变化为 {e['holdout_auc']-float(main_lgb.holdout_auc):+.5f}。
这是固定 seed、固定一次拆分下的结果，不是统计显著性检验。

## 怎样避免验证泄漏
所有实验使用同一 80/20 分层拆分，种子 42。
三个主表模型和历史增强版均使用开发集三折分层 CV；填补、编码、标准化仅在训练折拟合。
主表和增强版 LightGBM 使用相同的树数、学习率及参数网格，方便比较历史特征的作用。
历史聚合只用提供的历史记录和 ID，不用 TARGET；假设竞赛历史记录在申请时已可用，并未实施另一个真实时间截断实验。
按 CV 选择模型，按开发集 OOF 最大 F1 选阈值；holdout 不用于调参和选阈值。
CV 和 OOF 被用于选择，因此有乐观偏差；holdout 作为独立评估。不能再根据这份 holdout 报告继续调参。

## 增强版阈值与误判
开发集阈值 {e['threshold_from_development_oof']:.5f}；holdout precision={e['precision']:.4f}，recall={e['recall']:.4f}，F1={e['f1']:.4f}。

{groups.to_markdown(floatfmt='.4f')}

false_negative 是实际还款困难却未检出，false_positive 是没有该标签却被判为高风险。
这些均值只是错误诊断，可能受极端值与混杂因素影响；不代表因果关系或客户群体的固有风险。
最大 F1 的阈值并未纳入放贷利润或违约损失，不能直接当业务决策阈值。

## 图表与解释
![ROC comparison](output/{enhanced.name}/roc_curve.png)
![PR comparison](output/{enhanced.name}/pr_curve.png)
![Confusion matrix](output/{enhanced.name}/confusion_matrix.png)
![SHAP summary](output/{enhanced.name}/shap_summary.png)

增强版 LightGBM 在开发集抽样上的 mean absolute SHAP 前十名：
{important.to_markdown(floatfmt='.5f')}

SHAP 横轴解释树模型的原始输出贡献，不是概率百分点；红色表示特征值高，蓝色表示低。
平均绝对值大表示模型更依赖该特征，不说明其因果效应。
主表模型的 permutation importance 另见 output/real；两类重要性尺度不同，不应直接比较数字大小。

## 提交文件与可复现性
output/submission.csv 包含 48,744 行，列为 SK_ID_CURR、TARGET（概率）。
模型按开发集 CV 选定配置后，使用全部 307,511 行训练标签重新拟合。
测试集没有真值，**没有测试集 AUC，也没有 Kaggle 线上得分**。
输入 SHA256 见 data_manifest.json，运行环境见 requirements-tested.txt；原始 CSV 不打包、不上传 GitHub。

## 项目限制
新增表按记录聚合：分期部分付款记录可能重复计划金额，付款比例和日期差按付款事件计算，不代表每期最终还清比例；月度余额之和也不是当前未偿余额。新增表未完整编码类别字段。
历史拖欠比例的分母包含已关闭和未知状态月份，历史更长的贷款权重更大。无信用局记录者的计数字段目前采用中位数填补加缺失指示，未显式全部设零。关键字段口径见 FEATURE_GUIDE_ZH.md。
仅做小范围网格搜索；没有嵌套验证或多个随机种子的稳健性实验。
{extra_note}
随机 holdout 不能证明未来时间段表现；未做概率校准、业务成本优化和上线监控。

## 文件入口
- README.md：安装、运行、GitHub 上传说明。
- main.py：三模型主表基准。
- history.py：分块历史聚合。
- enhanced.py：bureau 中间版、完整历史版 LightGBM 与 CV 选择。
- extra_history.py：四张新增历史表的分块数值聚合。
- walkthrough.ipynb：逐步查看结果。
- output/real/RESULTS_ZH.md：主表阶段的单独报告。
'''
    (root/'RESULTS_ZH.md').write_text(overview)
    print('Overall report written:',root/'RESULTS_ZH.md')
