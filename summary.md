# 原始 7,000 Demo / 2,000 Meta / 8-shot 官方Test补充评估

原始TabPFN与固定Similar的8-shot Test Accuracy相同，没有准确率提升。
TabPFN−Similar=+0.000pp（+0条），95% paired bootstrap CI=[0.000, 0.000]pp；exact McNemar p=1。

## 配置与复用

完整沿用原始Phase 1–3：Demo7000、Meta2000、Top32 cosine、k8、S/D/B及MMR λ=.5；all-MiniLM-L6-v2冻结嵌入；原四特征与likelihood元标签。只加载results/phase3/models原LR与TabPFN-v2，不重训、不调参、不校准，不使用4-shot划分/模型/特征。两模型原1003条Validation策略与概率重放核验通过。
Similar的3080条预测来自此前已完成的7000池/8-shot历史基线缓存，逐ID、候选、8个示例、prompt hash、checkpoint和parser校验后原样复制；本轮Similar生成调用数为0。仅新增3080×2=6160条Diverse/Balanced greedy预测。Qwen revision与tokenizer均为989aa7980e4cf806f80c7fef2b1adb7bc71aa306，FP16/MPS、原模板、原greedy和strict parser。
Test gold只在全部预测及原模型策略决策封存后连接计算指标。该官方Test此前已在项目中评估过，本次是冻结原8-shot方案的补全，不是新获取的未使用测试集；不依据此次结果改模型或筛选参数。

## 官方Test结果（3080条）

| Method | 正确数 | Accuracy | Macro-F1（77类） | Invalid | S/D/B | 有益/有害/中性切换 | ΔSimilar(pp) |
|---|---:|---:|---:|---:|---|---|---:|
| always_similar | 2736 | 88.831% | 0.88964 | 1.429% | 3080/0/0 | 0/0/0 | +0.000 |
| always_diverse | 2708 | 87.922% | 0.87946 | 1.169% | 0/3080/0 | 67/95/2918 | -0.909 |
| always_balanced | 2740 | 88.961% | 0.89136 | 1.429% | 0/0/3080 | 63/59/2958 | +0.130 |
| logistic_regression | 2733 | 88.734% | 0.88841 | 1.461% | 1208/1502/370 | 19/22/1831 | -0.097 |
| tabpfn_v2 | 2736 | 88.831% | 0.88964 | 1.429% | 3080/0/0 | 0/0/0 | +0.000 |
| Oracle†（事后上界，不可部署） | 2826 | 91.753% | 0.91923 | 0.844% | 2990/27/63 | 90/0/0 | +2.922 |

† Oracle仅为8-shot三策略的事后正确性上界，不可部署。附加F1/invalid/混淆统计对应预声明gold-assisted见证：正确S优先，其次B、D，全失败则S；这些不是Oracle固有指标。invalid视为错误/FN，Macro-F1覆盖全部77类；混淆另存77×77、invalid向量和77×78扩展版，不漏计无效输出。

## 切换与Oracle

Similar错误=344；其中90条可挽救。Oracle=2826/3080（91.753%），相对Similar余量=2.922pp。
- logistic_regression：S/D/B=1208/1502/370；有益/有害/中性=19/22/1831；净正确数=-3；Oracle净余量兑现=-3.333%；Oracle gap=3.019pp。
- tabpfn_v2：S/D/B=3080/0/0；有益/有害/中性=0/0/0；净正确数=+0；Oracle净余量兑现=0.000%；Oracle gap=2.922pp。

## 配对统计

| Comparison | ΔAccuracy(pp) | 95% paired CI(pp) | exact McNemar p | Holm p |
|---|---:|---:|---:|---:|
| tabpfn_v2 − always_similar | +0.000 | [0.000, 0.000] | 1 | 1 |
| logistic_regression − always_similar | -0.097 | [-0.519, 0.325] | 0.755229 | 1 |
| tabpfn_v2 − logistic_regression | +0.097 | [-0.325, 0.519] | 0.755229 | 1 |
| always_diverse − always_similar | -0.909 | [-1.721, -0.097] | 0.0335723 | 0.167861 |
| always_balanced − always_similar | +0.130 | [-0.584, 0.844] | 0.786058 | 1 |
| always_diverse − always_balanced | -1.039 | [-1.753, -0.357] | 0.00350468 | 0.0210281 |
| oracle_retrospective − always_similar | +2.922 | [2.338, 3.506] | 1.61559e-27 | 1.13091e-26 |

5000次查询级paired bootstrap、seed42；同时保存Accuracy/Macro-F1及其配对差值CI。exact McNemar为双侧，主要比较TabPFN−Similar；另给预声明家族的Holm校正。Oracle检验只描述上界。Test已有历史使用，统计结果不作为反复调参后独立确认的替代。

## 原始Validation对照

| Method | Validation Accuracy | Test Accuracy | Test−Validation(pp) | Validation Macro-F1 | Test Macro-F1 |
|---|---:|---:|---:|---:|---:|
| always_similar | 87.836% | 88.831% | +0.995 | 0.87700 | 0.88964 |
| always_diverse | 85.643% | 87.922% | +2.279 | 0.85648 | 0.87946 |
| always_balanced | 86.939% | 88.961% | +2.022 | 0.86976 | 0.89136 |
| logistic_regression | 87.238% | 88.734% | +1.495 | 0.87156 | 0.88841 |
| tabpfn_v2 | 87.836% | 88.831% | +0.995 | 0.87700 | 0.88964 |
| oracle_retrospective | 89.831% | 91.753% | +1.923 | — | 0.91923 |

两划分互不重叠：Validation1003条、官方Test3080条且每类40条。跨划分差值为点估计，不进行配对推断；样本组成与既有Validation探索均可能影响差距。

## 结论与文件

原始TabPFN与固定Similar的8-shot Test Accuracy相同，没有准确率提升。
结果仅适用于原始冻结Banking77/Qwen/8-shot配置；收益受策略互补余量和有害切换限制。四维几何/标签结构统计未必足以识别最佳上下文。不得依据此Test继续训练、选择阈值或替换模型。
全部新文件在本目录。run_8shot_test.sh使用现有.venv与MPS执行；LR为CPU。test_metrics.csv、paired_comparisons.csv、validation_vs_test.csv保存完整数值；confusion_matrices/figures保存图表；execution_resources.json记录设备、耗时、内存（采样MPS分配非GPU峰值）。verification.json核对原文件未改、模型未训及Similar零重复推理。
