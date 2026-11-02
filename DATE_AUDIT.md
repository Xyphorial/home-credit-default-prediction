# 历史时间字段核对

已按块扫描全部四张新增历史表，核对相对申请日的日期和月份：

| 表 | 检查行数 | 正的实际事件/快照日期 | 缺失申请人 ID |
|---|---:|---:|---:|
| installments_payments | 13,605,401 | 0 | 0 |
| previous_application | 1,670,214 | 0 | 0 |
| credit_card_balance | 3,840,312 | 0 | 0 |
| POS_CASH_balance | 10,001,358 | 0 | 0 |

installments 的 DAYS_ENTRY_PAYMENT 与 DAYS_INSTALMENT 最大值为 -1；实际付款日期有 2,905 个缺失值。
信用卡和 POS 的 MONTHS_BALANCE 范围为 -96 至 -1。
previous 的 SK_ID_PREV 唯一。日期检查排除了 365243 异常占位值；模型也将其替换为缺失值。

DAYS_LAST_DUE_1ST_VERSION 中有 224,392 个正数，是原定到期日，属于计划信息，不自动等于未来已实现事件泄漏。
本检查没有发现正的实际付款/事件日期；这不替代数据来源在真实业务中可用时点的完整审计。
