# HWU64：沿用 Banking77 原方法的 8-shot Query-Adaptive Demonstration Selection

这是原 Banking77 项目的 **HWU64 第二数据集版本**。目的不是发明一套新 meta-label，也不是换模型，而是在另一套多领域意图分类数据上，按原项目方法检验 query-adaptive demonstration selection 是否仍然有效。

> **重要：本版本保留原方法。** 不使用“策略预测是否分类正确”的 0/1 correctness target；不要使用之前的 `HWU64_8shot_Code` 或 `HWU64_Selector_Fix_v2` 作为最终实验版本。此版本的 meta-label 与 Banking77 原报告一致：对每条 Meta-training query，在三种策略对应的 prompt 下，对该 query 的 Gold Intent continuation 计算 Qwen 的平均 token log-likelihood，选择分数最高的策略。评分不包含 prompt token 或 EOS。

## 1. 固定方法（和 Banking77 对齐）

| 环节 | 固定设置 |
|---|---|
| 数据集 | `DeepPavlov/hwu64`，64 个 intent；官方 train 8,954、test 1,076 |
| 划分 | 官方 train 按 seed 42 分为 Demo Pool 70%、Meta-training 20%、Validation 10%；官方 test 保持不动 |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2`，L2-normalized |
| Retrieval | cosine Top-32；排除相同 query ID 和相同规范化文本 |
| 8-shot 策略 | Similar / Diverse（greedy MaxMin）/ Balanced（MMR，λ=0.5） |
| 分类 LLM | 冻结 `Qwen2.5-1.5B-Instruct`，greedy decoding，不微调 |
| Meta-features | Top-32 最大相似度、平均相似度、标签分布熵、最大标签投票比例（4 维） |
| Meta-label | Qwen 对 Gold Intent continuation 的 mean token log-likelihood；选最高分；容差 1e-6；平局 Similar > Balanced > Diverse |
| Meta-selector | Logistic Regression（StandardScaler + stratified CV / macro-F1 网格选择）和 TabPFN-v2（原 v2 checkpoint，4 estimators） |
| 评估 | Accuracy、64 类 Macro-F1、Invalid、策略切换/Oracle、5,000 次 paired bootstrap、Exact McNemar、Holm 校正 |

只有数据集和按官方 train 规模比例生成的 partition 数量不同。模型、四个 meta-features、selection strategies、meta-label 产生方式和 selector 类型均沿用 Banking77。实际数据集 revision、模型 revision、软件版本和硬件会记录在输出文件中。

### 研究问题与组员顾虑的关系

HWU64 是第二个数据集。它能帮助检验方法是否不仅适用于 Banking77 银行业务域，也能应用于 personal assistant / smart-home 场景的多领域意图分类。这是对**跨数据集适用性/泛化能力**的补充证据。

必须准确表述：由于按要求保留 Qwen log-likelihood meta-label，这个扩展本身并没有消除“meta-label 由 LLM 分数构造，因此标签创新性有限”的概念性顾虑。它回应的是方法能否跨数据集复用，而不是把原监督目标包装成全新算法。报告会明确描述方法和局限，不能声称已经解决所有创新性争议。

## 2. Mac 从零开始

推荐 Python 3.12。**不要继续使用你之前 Python 3.9 的 `.venv`**：TabPFN 9.2.0 要求 Python 3.10 或以上。建议先新建 `.venv312`，旧 `.venv` 暂时保留，避免误删。

### Step 1 - 安装 Python 3.12

在 Terminal 检查 Homebrew：

```bash
brew --version
```

有 Homebrew 就安装：

```bash
brew install python@3.12
```

确认：

```bash
"$(brew --prefix python@3.12)/bin/python3.12" --version
```

如果没有 Homebrew，可从 Python 官方 macOS 下载页安装 Python 3.12：<https://www.python.org/downloads/macos/>。

### Step 2 - 进入本项目文件夹

```bash
cd ~/Downloads/HWU64_8shot_OfficialMethod
```

如果你把文件夹放在其他位置，就按实际路径修改 `cd` 后的路径。

### Step 3 - 新建 Python 3.12 环境

```bash
"$(brew --prefix python@3.12)/bin/python3.12" -m venv .venv312
source .venv312/bin/activate
python --version
python -m pip install --upgrade pip setuptools wheel
```

必须看到 `Python 3.12.x`。之后每次重新打开 Terminal，都先进入项目目录，再运行：

```bash
source .venv312/bin/activate
```

### Step 4 - 安装依赖

```bash
python -m pip install -r requirements.txt
```

首次安装会下载 PyTorch、Transformers、TabPFN 等包，可能需要一些时间。

### Step 5 - 检查环境和代码

```bash
python scripts/check_environment.py
python -m pytest -q
```

测试应全部通过。Apple Silicon Mac 应显示 `MPS available: True`。如果是 `False`，不要贸然运行完整实验，先把检查输出发给组员/助教确认设备兼容性。

### Step 6 - 运行完整实验

```bash
python run.py all
```

程序按以下顺序自动执行：

1. 下载固定数据版本并建立 train/test；
2. 将 train 分成 demonstration pool、meta-training、validation；
3. 生成 MiniLM embeddings；
4. 为 query 检索 Top-32，建立三套 8-shot demonstrations 和四维 meta-features；
5. 用冻结 Qwen 对 Validation/Test 三种策略生成分类结果；
6. 对 Meta-training 的每个 Gold Intent，在三种策略 prompt 下计算 Qwen mean token log-likelihood，构造 meta-label；
7. 训练 LR 和 TabPFN-v2 selector；
8. 评估固定策略、自适应策略、Oracle 和统计检验；
9. 生成 Markdown 和 PDF 报告。

## 3. 分阶段运行（可续跑）

如需分开运行或出错后恢复，依次运行：

```bash
python run.py prepare-data
python run.py embed
python run.py prepare-queries
python run.py predict
python run.py evaluate
```

`evaluate` 会自动完成 Meta-training likelihood scoring、meta-label 构造、LR/TabPFN 训练、评估和报告生成。如果中途停止，重新运行同一阶段即可；已完成并通过身份校验的 embeddings、Qwen predictions 和 likelihood scores 会使用缓存。

只重新生成已有结果的报告：

```bash
python run.py report
```

注意：`report` 只生成 PDF/Markdown，不会替你执行缺少的实验阶段。它要求 `results/evaluation.json`、`results/meta_label_diagnostics.json` 和 split manifest 已存在。

## 4. 预计计算量

按 70/20/10 划分，通常为 Demo Pool 6,268、Meta-training 1,791、Validation 895、官方 Test 1,076。Qwen 主要工作量约为：

- Validation + Test：`(895 + 1,076) × 3 = 5,913` 次 greedy generation；
- Meta-label scoring：`1,791 × 3 = 5,373` 次 teacher-forced likelihood scoring；
- 合计约 11,286 次策略级 Qwen 操作。

实际运行时间取决于 Mac 型号、内存、模型下载速度和设备后端。首次需要下载 HWU64、MiniLM、Qwen、TabPFN-v2 checkpoint。TabPFN checkpoint 可能要求先接受 Prior Labs 的模型许可条款。

## 5. 最终文件

主要报告：

```text
report/HWU64_实验报告.pdf
report/summary.md
```

核心结果和审计文件：

| 路径 | 内容 |
|---|---|
| `data/splits/manifest.json` | 数据集 revision、split size、seed 和标签映射 |
| `cache/queries/` | Top-32、四维 meta-features、每种策略 8-shot demonstrations |
| `cache/predictions/` | Validation/Test 三种策略的 Qwen 原始输出和解析结果 |
| `cache/scores/meta_training_likelihood/` | Meta-training 每条 query / strategy 的 mean token log-likelihood |
| `results/meta_training_dataset.csv` | 四维 meta-feature、三种 likelihood 分数和最终 meta-label |
| `results/meta_label_diagnostics.json` | meta-label 分布、margin、并列统计 |
| `results/evaluation.json` | Accuracy、Macro-F1、Invalid、Oracle、paired tests、切换分析 |
| `results/*_predictions_and_choices.csv` | 每条 Validation/Test query 的真实标签、预测和选择器决策 |

## 6. 重要研究纪律

- 不得为了好看而改动结果、筛选 test query、重采样 test、平衡 test meta-label 或根据 Test 表现调参。
- LR 和 TabPFN 只在 Meta-training labels 上拟合；Validation/Test 只作评估。
- Oracle 根据 gold label 事后选策略，是不可部署的理论上界。
- likelihood meta-label 不等同于最终 greedy 预测的 0/1 correctness；必须报告 meta-label 分布和 margin。
- 如果 selector 仍然偏向某个固定策略，或者没有超过最强固定策略，这也是有效的实验结果。必须如实讨论。
- Banking77 的数值不可复制到 HWU64；新报告只使用本次 HWU64 实测结果。

## 7. 数据与模型来源

- HWU64 dataset: <https://huggingface.co/datasets/DeepPavlov/hwu64>
- Original HWU64 source: <https://github.com/jianguoz/Few-Shot-Intent-Detection>
- Sentence encoder: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- Qwen: <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct>
- TabPFN-v2 checkpoint: <https://huggingface.co/Prior-Labs/TabPFN-v2-clf>
