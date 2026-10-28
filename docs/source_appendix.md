# 完整源码与附录对应

本版本附录A列出支撑材料，附录B说明运行环境，附录C逐文件刊载全部23个Python文件及运行配置原文，包括空的包初始化文件。新增的两个程序只负责生成、核验附件5结果表，不改变模型算法、参数或冻结实验结果。

在压缩包根目录运行：

```
python -m pip install -r requirements.txt
python run_final_model.py
```

快速统计核验：python run_final_model.py --verify-only

单日实际求解：python run_final_model.py --smoke --with-reference

无参数命令执行五种子334天回放，输出reproduced/。运行`python source_code/export_submission_excels.py`重建5份官方模板结果，运行`python source_code/validate_submission.py`核验提交副本。现有5份xlsx均为结果文件，不是配置。未使用SPSS交互命令或Excel宏。预测来源与原始Excel数据处理如单独调用，另安装lightgbm并提供赛题原始附件；这不属于默认冻结预测回放。
