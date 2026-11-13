# CLINC150 四特征 8 shot 自适应 ICL 实验

[最终中文报告](results/summary.md) · [Test指标](results/test_metrics.csv) · [Validation指标](results/validation_metrics.csv) · [配对统计](results/paired_comparisons.csv) · [验证记录](results/verification.json)

这次实验已真实完成：15,000次元训练似然评分，Validation 9,000条及Test 13,500条固定策略预测。LR和TabPFN-v2均使用CLINC150数据重新拟合，没有加载Banking77已训练选择器。

| 方法 | Validation Accuracy | Test Accuracy | Test正确数 |
|---|---:|---:|---:|
| Similar | 91.667% | 91.067% | 4098/4500 |
| Diverse | 92.200% | 91.044% | 4097/4500 |
| Balanced | **92.467%** | **91.533%** | 4119/4500 |
| LR Adaptive | 92.333% | 91.267% | 4107/4500 |
| TabPFN-v2 Adaptive | 91.833% | 91.333% | 4110/4500 |
| Oracle 事后上界 | 94.367% | 94.133% | 4236/4500 |

**结论：TabPFN会切换，但未确认优于固定策略的可靠优势。** 它在Test选择2789次Similar和1711次Balanced；相对Similar有26次有益、14次有害切换，净增12条（+0.267pp），精确McNemar p=0.08069，Holm p=0.72621。Validation预选的固定Balanced仍多答对9条。Oracle不可部署。

## 目录

- `results/`：完整中文报告、全部逐题预测/评分、元训练表、选择器决策、配对统计、逐类分析、5张图及原运行审计证据。
- `provenance/`：原选择器训练元数据；可追溯到固定checkpoint、四特征及重载核验。
- `reference/banking77_8shot/`：报告采用的Banking77对照指标与说明。
- `reproduction/`：独立复现项目，包括逐字保留的实验实现、共享模块、配置、原始数据/划分和测试。新运行输出写在其下的`experiments/clinc150_8shot/results/`，不会覆盖本目录的已发布结果。
- `PACKAGING_MANIFEST.json`：每个拷贝文件的原件和发布件SHA256；仅报告的图像链接及发布说明经过展示适配，数值文件保持原样。

## 只核验结果 不运行模型

使用已有numpy/pandas/scipy/scikit-learn环境，在仓库根目录运行：

```bash
python CLINC150_8shot/verify_results.py
```

如需创建轻量环境：

```bash
python3 -m venv .venv-review
.venv-review/bin/python -m pip install -r CLINC150_8shot/requirements-review.txt
.venv-review/bin/python CLINC150_8shot/verify_results.py
```

脚本核验打包hash、15,000条似然记录及目标、22,500条固定预测、两个split共12行Accuracy/Macro-F1、切换计数、Oracle及McNemar/Holm结果，不下载模型，也不调用LLM。bootstrap样本未上传，5000次配对bootstrap的结果及固定seed已保存。

## 从头复现实验

使用支持原生MPS的Apple Silicon环境。保持本实验MPS配置，不自动回退到CPU运行LLM。先进入独立复现项目：

```bash
cd CLINC150_8shot/reproduction
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/download_resources.py
.venv/bin/python -m pytest -q tests experiments/clinc150_8shot/test_clinc150.py
bash experiments/clinc150_8shot/run_experiment.sh
.venv/bin/python experiments/clinc150_8shot/results/finish_report.py
```

`requirements-phase3-lock.txt`记录原环境依赖，实际关键版本另见已发布`results/execution_environment.json`。全量复现需要下载固定Qwen、MiniLM和TabPFN checkpoint，并执行数小时MPS推理。模型权重、已训练模型二进制、嵌入数组及逐操作缓存未打包；中断后同命令支持续跑。

下载脚本仅为本checkout解析资源路径，不改变模型版本、提示、选择策略、四特征或统计协议。实验核心`.py`、原始共享模块和测试按原件拷贝，可用manifest核对。复现生成新的冻结协议和审计记录；不能把原运行`frozen_protocol.json`直接当作另一台机器已有全部缓存的证明。

## 方法与数据来源

仅使用官方`data_full.json`的150个in-scope类别，排除OOS；Demo=10000、Meta=5000、官方Validation=3000、Test=4500，seed42。Qwen2.5-1.5B-Instruct冻结、MiniLM检索Top32、k=8、S/D/B、MMR λ=.5；元标签为gold-label平均token log-likelihood，原四特征，原LR/TabPFN-v2方法。模型与决策在Test评价前冻结，固定Balanced由Validation选择。

CLINC150数据来源于[Larson等人的官方仓库](https://github.com/clinc/oos-eval/tree/828f8093932c8fe6ca7936c3d2e52903b1c523de)，对应EMNLP 2019论文 *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*。原始数据及其派生划分保留[CC BY 3.0许可证](reference/CLINC150_LICENSE.txt)；文本和标签未改写，仅划分训练集并添加稳定ID/整数映射。数据中存在官方跨分区重叠，报告已披露，未删改Test以筛选结果。

原Banking77/HWU64资料保留在仓库原位置；不要把根目录历史`results/`与这里的CLINC150结果混用。
