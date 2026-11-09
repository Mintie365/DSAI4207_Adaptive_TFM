# HWU64 改进版：从这里开始

本版本只修复运行入口、环境检查和启动脚本；**不改变实验方法、数据划分、模型、提示词、特征或评价指标**。

## 1. 解压并进入项目目录

将 ZIP 解压到 Downloads，打开 Terminal：

```bash
cd ~/Downloads/HWU64_8shot_Improved
```

你之前能用的 Python 环境在旧目录 `~/Downloads/HWU64_8shot_OfficialMethod/.venv312`。**不用重新安装，也不要移动旧环境**；下面通过 `PYTHON=...` 直接调用它。

如果解压后的文件夹名称不同，先在 Finder 中确认名称，再把上面的路径改成实际路径。

## 2. 激活你已经建立的 Python 3.12 环境

```bash
PYTHON="$HOME/Downloads/HWU64_8shot_OfficialMethod/.venv312/bin/python" "$HOME/Downloads/HWU64_8shot_OfficialMethod/.venv312/bin/python" --version
```

应显示 Python 3.12.x。

应显示 Python 3.12.x。若项目目录内没有 `.venv312`，不要重复创建环境；把你原来可用的 `.venv312` 文件夹复制到这个项目目录，或把代码文件夹放回原项目目录中。

## 3. 检查环境

```bash
PYTHON="$HOME/Downloads/HWU64_8shot_OfficialMethod/.venv312/bin/python" "$HOME/Downloads/HWU64_8shot_OfficialMethod/.venv312/bin/python" scripts/check_environment.py
```

本版本已修复 `ModuleNotFoundError: No module named 'src'` 的检查脚本路径问题。

## 4. 运行离线测试

```bash
PYTHON="$HOME/Downloads/HWU64_8shot_OfficialMethod/.venv312/bin/python" "$HOME/Downloads/HWU64_8shot_OfficialMethod/.venv312/bin/python" -m pytest -q
```

必须看到所有测试通过后再继续。

## 5. 检查 Hugging Face 连接

```bash
curl -I --connect-timeout 10 --max-time 20 https://huggingface.co
```

如果这里超时，说明当前网络无法连接 Hugging Face。增加 Python 下载超时时间不能修复网络阻断；先解决网络连接，不要反复重跑整个实验。

## 6. 运行完整实验

```bash
PYTHON="$HOME/Downloads/HWU64_8shot_OfficialMethod/.venv312/bin/python" bash scripts/run_hwu64.sh all
```

脚本默认优先使用 `.venv312`，自动设置项目导入路径，并将 Hugging Face 元数据超时设为 15 秒、下载超时设为 180 秒。可用已有缓存续跑；未下载的模型/数据仍需要网络。

## 7. 如果中途停止，按阶段继续

只运行尚未完成的阶段，不要每次都从头开始：

```bash
PYTHON="$HOME/Downloads/HWU64_8shot_OfficialMethod/.venv312/bin/python" bash scripts/run_hwu64.sh prepare-data
PYTHON="$HOME/Downloads/HWU64_8shot_OfficialMethod/.venv312/bin/python" bash scripts/run_hwu64.sh embed
PYTHON="$HOME/Downloads/HWU64_8shot_OfficialMethod/.venv312/bin/python" bash scripts/run_hwu64.sh prepare-queries
PYTHON="$HOME/Downloads/HWU64_8shot_OfficialMethod/.venv312/bin/python" bash scripts/run_hwu64.sh predict
PYTHON="$HOME/Downloads/HWU64_8shot_OfficialMethod/.venv312/bin/python" bash scripts/run_hwu64.sh score-meta
PYTHON="$HOME/Downloads/HWU64_8shot_OfficialMethod/.venv312/bin/python" bash scripts/run_hwu64.sh evaluate
```

全部完成后生成报告：

```bash
open report/HWU64_实验报告.pdf
```

若结果文件已齐全、只需要重新生成 PDF：

```bash
PYTHON="$HOME/Downloads/HWU64_8shot_OfficialMethod/.venv312/bin/python" bash scripts/run_hwu64.sh report
```

## 重要说明

- 这不是更换方法的版本。仍然使用 Qwen Gold-label mean token log-likelihood 构造 meta-label；prompt/EOS 不计入；平分顺序为 Similar > Balanced > Diverse。
- 仍然使用 Similar、Diverse（Greedy MaxMin）、Balanced（MMR，lambda=0.5），Top-32、8-shot、四个原定 meta-features、Logistic Regression 和 TabPFN-v2。
- 仍然使用原定评估和统计检验。
- 本次修复的是启动和环境检查，不会凭空消除 Hugging Face 网络故障，也不会伪造实验结果。
