# 从零复现指南

1. 用支持标准ZIP-LZMA的WinRAR或7-Zip完整解压，保留最外层support_material_q2_final目录。Python标准库zipfile也支持解压。
2. 安装Python 3.12，按README创建虚拟环境，安装source_code/optimization_v5/requirements.txt。依赖版本是实际验证版本，安装需要可访问相应包索引；本次未在空白联网环境重新安装依赖。
3. 在support_material_q2_final根目录运行：

```shell
python run_final_model.py
```

默认执行种子1—5的完整334天回放，不需要个人路径、外部工程或商业求解器。冻结预测已经包含在data内，不训练模型。每个RUN/DONE输出指出正在处理的问题与种子。若想先核对安装和各求解分支：

```shell
python run_final_model.py --smoke
python run_final_model.py --verify-only
```

smoke仅求解2025-02-01，各随机问题默认种子1，不代表全年验证；verify-only读取归档逐日数据重新核算20组结果、五种子统计和10000次Bootstrap，同时重解Q1。二者均不会改results。

4. 查看输出：

| 问题 | 归档结果 | 新运行结果 |
| --- | --- | --- |
| Q1 | result/Q1/Q1_final_result.csv、Q1_final_trajectories.csv | reproduced/Q1_summary.csv、Q1_trajectory.csv |
| Q2 | result/Q2/Q2_final_result.csv及q2_seed*/daily.csv | reproduced/checkpoints/q2__joint_50__seed*/ |
| Q3 | result/Q3/Q3_final_result.csv、Q3_paired_summary.csv、Q3_bootstrap.json | reproduced/checkpoints/q3__baseline__seed*/ |
| Q4日前 | result/Q4/Q4_day_ahead_final_result.csv | reproduced/checkpoints/q4_2__cluster__seed*/ |
| Q4日内 | result/Q4/Q4_intraday_final_result.csv | reproduced/checkpoints/q4_3__cluster__seed*/ |

reproduced/final_replay.csv是本次真正重求解的汇总；requested_replay_statistics.csv按本次请求的种子计算均值，只有完整默认五种子才是论文主口径。five_seed_recomputed.csv和five_seed_statistics.csv是对归档五种子逐日数据的独立重算；文件名称故意区分，不能混用。

5. 部分运行及恢复：`python run_final_model.py --question q4_3 --seed 1`只跑所选组合。再次相同命令会校验检查点摘要哈希并恢复，不能修改参数后沿用旧缓存。不要同时启动两条写入同一任务目录的命令。异常中断后保留错误状态和writer.lock；确认原进程已结束后，才可移走该任务目录并从空目录重跑。

6. Q3配对收益相对于独立连续SOC的固定V2参考，既不是Q2，也不是论文旧版单种子。其五种子逐日记录在fixed_reference_seed*，verify-only逐日相减验证Q3_paired_daily.csv后重算Bootstrap。7日非循环移动块、10000次、随机种子20260912，五个模型种子共享重采样日期块。如需从冻结输入重新求解配对参考，运行`python run_final_model.py --question q3 --with-reference`；新参考写入reproduced/reference_checkpoints，核对归档参考费用。它是验证对照，不能替换正式Q3。

7. 下界、消融、风险网格均为附加证据。不要将代表日诊断当作全年经济指标或新的选参结果；下界固定各实际日初/日末SOC，滚动方案还固定零点合同和已承诺前六小时。配置不改变，所有新结果若与归档差异超阈值，入口会明确失败而非覆盖归档。

8. 官方结果表使用赛题附件5原模板。论文统计量取五种子均值，而必须提交的一条完整购电策略采用种子1逐槽结果。重新生成与核验命令如下，默认写入`reproduced/submission_excels/`，不会覆盖`result/`中的提交副本：

```shell
python source_code/export_submission_excels.py
python source_code/validate_submission.py
python source_code/validate_submission.py --result-dir reproduced/submission_excels --report validation/reproduced_submission_excel_checks.json
```

结果表的时间标签、费用口径和字段映射见`docs/submission_excel_notes.md`。
