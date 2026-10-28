# 当前版本修正与验证范围

本版补齐官方结果模板、结果表生成与核验程序及附录索引；模型Python源码、数学参数、冻结CSV输入、CSV结果及历史实验数据保持不变。

## 本版修正

1. 论文附录A与file_list.md补列AI工具使用详情.pdf，并列入本说明。
2. AI详情的图表数据口径改为各问最终采用结果：问题二为更新结果，其余问题沿用最终确认的v5结果。
3. reproduce_guide.md采用真实根目录support_material_q2_final。
4. 五份Excel中的来源路径由results/统一为result/，结果值保持不变。
5. 本说明区分本版验证与历史档案，不改写历史报告中的通过结论或实验数字。
6. 以附件5原始模板重建5份结果表，写入48,096个年度计划槽、需要时的48,096个调整槽、334天分段充放电/SOC及全部紧急购电连续时段。
7. 新增`export_submission_excels.py`和`validate_submission.py`，提交副本及独立再生副本均通过27项结构与数值检查。

## 验证范围

在本版独立运行副本中执行python run_final_model.py --verify-only及python run_final_model.py --smoke --with-reference。前者重解Q1、重算20组334天归档统计与Q3 Bootstrap；后者检查2025-02-01首日求解及固定参考。检查源码与论文附录对应、文件清单、包内完整性哈希、Excel单元格值，以及修正前后论文附录之前的内容和渲染页面。两条运行命令均通过；Excel的41300个数值单元格保持一致，仅7个来源说明单元格修改路径。文件清单覆盖177个文件，论文附录源码与配置对应检查通过。

本版未重新训练预测模型，未重新执行全部334天×所有种子的优化，也未重新执行全部消融及历史诊断实验；不将归档重算或首日检查表述为全年重新求解。

## 历史档案的解释

- validation/final_checks.json：历史98页源码恢复版检查；其中docx_bytes、pdf_bytes及像素比较字段属于当时对象，不适用于本版文件大小。当前文件指纹以本版交付清单为准。
- validation/verification.json：历史35组、11690策略日验证，保留原始脚本名和指标；不是当前全部结果的新运行记录。
- validation/docx_scope_check.json、appendix_entry_checks.json、package_checks.json、statistics_recheck.json、xlsx_value_checks.json：此前阶段的检查记录，保留原文，不自动视为本版重验结果。
- validation/v5_*：历史v5对照、诊断或敏感性证据，须按各表策略和实验口径解释；不能替代当前Q2采用结果。
- verification_report.md：此前Q2同步阶段的验证说明，不作为本版重新完成全年实验的声明。
- validation/final_results.csv、result各问CSV：当前保留的正式结果，本轮只复算与核对，不改值。

source_code/config/source_manifest.json与package_integrity.json均保留原内容。本版修改的说明文件和Excel路径不在需要更新的摘要条目中；现有完整性条目全部核对一致，论文附录C与这些配置仍一致。
