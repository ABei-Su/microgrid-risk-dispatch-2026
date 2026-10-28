# 微网调度：Q2更新后的配套材料

本包采用Q2最终风险权重、压力概率和释放系数，其余保留v5代码、参数和预测输入复现微网Q1—Q4调度。原论文不包含在压缩包中。`result/result1.xlsx`至`result4-3.xlsx`均以赛题附件5原模板为底稿，填入最终策略的购电、调整、充放电和紧急购电明细。

| 问题 | 正式模型 | 主要实现 |
| --- | --- | --- |
| Q1 | 确定性词典序线性规划 | source_code/optimization_v5/solver.py:solve_q1 |
| Q2 | 50场景联合残差路径、两阶段随机LP、CVaR95 | source_code/src/scenarios.py、policies.py、solver.py |
| Q3 | 锁定月度预测选择、固定1元收益门槛、冻结前缀及连续SOC | source_code/optimization_v5/replay.py、run_experiments.py |
| Q4日前 | 联合场景压缩和共同场景安全筛选 | source_code/optimization_v5/policies.py、replay.py |
| Q4日内 | 联合场景压缩、固定门槛与日内改签 | source_code/optimization_v5/policies.py、replay.py |

## 环境与运行

实际验证环境：Python 3.12.14，NumPy 2.5.3，pandas 3.0.1，SciPy 1.18.1，openpyxl 3.1.5。求解使用SciPy内置HiGHS，单线程；openpyxl用于附件5结果表生成与校验。求解库版本记录见source_code/config/environment.json。

```shell
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_final_model.py
```

默认命令执行Q1典型日求解，以及Q2、Q3、Q4日前/日内各5种子334天回放。它读取包内冻结预测CSV，**不重新训练预测模型**。完整运行时间随CPU与求解器而变化，应预留较长时间；按块输出进度并支持相同包指纹下继续运行。快速核对可用`python run_final_model.py --verify-only`，该模式重算归档统计及Bootstrap并求解Q1，不能称为全年重新求解。

## 输入与输出

- data：实际观测、官方预报、典型日、冻结预测。为了体积限制，只保留正式回放读取的列；每一保留单元格的原始字符串和所有行顺序均未改变。source_code/config/input_projection.json逐列记录。
- source_code/optimization_v5/preprocessing：原数据读取、时间对齐、PCHIP插值和严格滞后特征源码；原始赛题附件由赛题提供，不重复打包。
- source_code/optimization_v5/forecast_provenance：产生冻结预测的核心训练/特征源码，作为来源说明；默认运行不调用。若单独研究这些函数，需要额外的LightGBM和原始赛题数据，不能将重训结果无核验地替换本包冻结输入。
- result/Q1—Q4：五种子年度统计、逐日费用、更新日志、求解日志及必要对照。
- result/result1.xlsx—result4-3.xlsx：附件5官方格式结果。随机问题采用种子1的一条完整可执行策略填表；论文主结果仍为五种子均值，两者口径在docs/submission_excel_notes.md中说明。
- data/templates：未填数值的附件5原始模板，供结果表再生和结构核验。
- validation：五种子均值/标准差/范围、条件下界、风险网格、概率和历史边界证据；paired_policy_comparisons.csv中的baseline仅是有明确标记的对照，不能当正式模型结果。
- figure/data：最终统计图所需数据与代表种子轨迹；不复制或修改论文图片。
- reproduced：运行时新建，存新求解和核验结果，不覆盖results。其checkpoints保存十分钟明细、概率/历史审计及跨日SOC。

Q1是一个确定性典型日，原334次×5的记录是重复验证，不能累计为年度费用。随机问题的费用和紧急量先按种子累计334天，CVaR先按种子的334个日费用计算，再取五种子均值；SD为样本标准差。单种子SOC用于代表展示。

代码出处与摘录范围见source_code/config/source_manifest.json。包装层只调整路径、完整性校验和输出目录，并删除不可达的实验分支；数学求解函数和生效参数不变。final_config.yaml保留原v5签名用于溯源，本包另用package_integrity.json检查打包副本，绝不伪造原签名。

## 完整源码附录

全部23个Python源码及配置已同步收入本版本论文附录C。除模型入口外，`source_code/export_submission_excels.py`可从冻结逐槽结果重建5份官方结果表，`source_code/validate_submission.py`逐单元格核对模板结构和数值。

## 本轮核验范围

Q2采用费用14,204,057.19元、紧急购电64,789.16 kWh、CVaR95 69,900.56元。原始五种子334天实验已完成；本轮复核归档统计、Q1与Q3 Bootstrap，并运行Q2首日入口与原记录对照，没有重新训练。validation中未更新的v5比较/核验档案属于历史证据，当前年度数值以validation/final_results.csv及result各问CSV为准，不将历史v5比较标为新Q2验证。

## 当前版本与历史验证记录

本版补齐附件5官方结果表、结果表生成与核验程序、运行说明和论文附录索引，不改变模型、参数及冻结结果。当前核验见validation/current_review.md；历史验证档案保留原内容，不代表本次重新执行了全部全年实验。
