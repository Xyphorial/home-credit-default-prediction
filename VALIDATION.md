# 真实数据验证记录

已完整运行全量真实数据：主表三模型（三折 CV + 网格搜索 + 独立留出集），bureau/bureau_balance 历史增强 LightGBM，以及最终全量重训和测试集预测。

- train：307,511 行；test：48,744 行；holdout：61,503 行；development OOF：246,008 行。
- 验证 train/test ID 不重复且不相交；OOF/holdout 无交集并完整覆盖 train。
- 验证输出标签与源文件按 ID 对齐；概率有限且在 [0,1]；CSV 顺序与测试集完全一致。
- 从保存预测重新计算 AUC、precision、recall、F1，与 metrics.json 一致。
- 验证最终 submission 确实来自开发集 CV AUC 选中的模型。
- 加载已保存模型，重新计算前 100 条测试预测，与最终 submission 一致。
- 检查了 EDA、ROC/PR、SHAP 等图表布局。
- 合成边界检查：未知类别、整列缺失、零分母、DAYS_EMPLOYED 异常、历史关联、拖欠比例、未关联记录排除、分块聚合一致性。

主表 LightGBM：CV AUC 0.760748，holdout AUC 0.764897。
bureau 历史增强：CV AUC 0.766034，holdout AUC 0.769054。
本次真实结果没有使用模拟数据。此前的 demo 结果不包含在当前交付包中。

全部十个文件已收齐并完成完整历史版训练：CV AUC 0.779202，holdout AUC 0.782244。
使用 470 个编码前特征；新增四表含 29,117,285 条记录（各表行数以 data_manifest.json 为准）。
完整模型的前 100 条预测通过重新汇总原始历史记录复现，最大绝对差约 9.7e-17。
最终 CSV 的列名和申请人 ID 顺序与 sample_submission 完全相同。

最终全量重训第一次触及 8 GiB 内存限制；已从开发集检查点恢复，释放中间模型、按列构建直方图并限制直方图缓存至 128 MB 后成功完成。CV 与 holdout 成绩来自原先的开发集选定模型，恢复时没有重新选择参数。
