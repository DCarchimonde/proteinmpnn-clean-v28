# V9 循环稳定性修复与 17×100 交付说明

> 历史说明：V9 不含 RMSD 优先排序和单体重算，不能满足当前 Task 2/3。
> 当前唯一推荐入口是 `run_v10_rmsd_aware_1700_and_monomer.sh`，完整合同见
> `V10_TASK2_TASK3_运行与验收.md`。

## 当前结论

- 旧 V8 不能作为 17×100 的发布模型。旧候选 6,964 条中只有 1,768 条在所有循环起点下保持一致的 `>0.6` 甲基化硬标注；其余 5,196 条会跨阈值。
- 旧 3AV 候选 4,080/4,080 的最高甲基化物理位点均为第 7 位。降低阈值不能解决这个问题：阈值降至 0.4 仍只有第 7 位，降至 0.3 也只有一条新增第 6 位。
- 根因不是简单的“输出列没转回去”，而是训练与部署不一致：旧训练每个循环起点只优化一个 decoder order，部署却对每个起点的全部 decoder orders 求平均；同时 V8 用跨起点平均概率作正式释放判据，把最差起点和跨阈值差异仅作为诊断，并允许单靶点点位高度集中仍然通过。
- 本分支把修复协议升级为 V9；旧 V8 候选不能混入 V9 的最终 1,700 条。

## V9 已修复的代码合同

1. 训练时对每个物理循环起点完整计算全部 `L` 个 decoder orders，形成 `L×L` 网格。
2. 每个起点先对 decoder orders 可微求均值，再映回物理残基；正样本采用跨起点最小概率，负样本采用跨起点最大概率。
3. 训练目标包含严格正权重的 worst-start BCE 和所有有效位点的跨起点 span² 一致性损失；训练、验证和部署温度统一为 `T=0.5`。
4. 平均概率只用于排序。正式甲基化释放要求逐位 `round(probability_min, 8) > 0.6`，并且所有循环起点的硬标注完全一致。
5. 生成、精确 base 打分、最终选择和独立回放均绑定模型、方案、输入和上游产物 SHA-256；运行器还会重新核对缓存成品的实际哈希、1700 行和 17×100 配额。缺列、文件被删改、矩阵尺寸不符、重算不一致或哈希不符都会失败。
6. 17 个靶点使用同一协议，不冻结旧的“已通过”候选，也不以补齐方式伪造配额。
7. 选择前按 marked sequence、natural sequence、forward-cyclic natural sequence、历史池和先前池去重；最终还要求跨靶点 natural/cyclic 唯一。
8. 每靶点同时审计全部甲基位点、主位点、mean argmax、min argmax 和甲基化残基。任一物理点位或任一甲基化残基占比超过 80%，整批禁止发给尚哥。
9. 最终 1,700 条用 batch size 1 重放甲基概率和精确 receptor-visible `L×L` cyclic-base 证据；只有逐行复算与选择文件完全一致才生成交付表和 FASTA。该重放复用相同评分实现，属于数值/批大小/持久化一致性核验，不冒充第二套独立算法验证。

## 尚待 GPU 实跑证明的部分

- V9 已修复训练与部署的循环网格不一致，并能在生成与选择阶段硬拦截旧 V8 的第7位塌缩；但当前只重训 expert heads，没有凭空证明新 checkpoint 已消除靶点级位点塌缩。只有 GPU 实跑后 17 个靶点全部通过候选覆盖率、跨起点稳定性和 `≤80%` 集中度门，才能说这次模型运行可用于 1700 条交付。
- 151条记录是 V3–V9 反复使用的内部开发审计，不是新的 blind outer test。它不再参与 epoch 选择或 checkpoint promotion；论文级泛化结论仍需从未用于调参/放行的 structure/scaffold-grouped outer set。
- cyclic-base floor 取当前靶点生成池的底部1%分位，只是弱异常值过滤，不是独立 base plausibility 校准。论文若使用该指标，必须按此名称和限制解释。
- 若 GPU 实跑仍出现位点集中或候选覆盖不足，流程会硬停。此时应解冻/微调 trunk 或改用循环感知分类器并重新做外层验证，禁止只增加抽样次数来掩盖问题。

## 一键运行

在带 CUDA PyTorch 的 AutoDL/Linux 环境中，从仓库根目录执行：

```bash
bash run_cyclic_stability_v9_1700.sh
```

附带输入的冻结 SHA-256：

- `train_serine_provenance_corrected.jsonl`：`98c73a832e3e46820018354ca50a378739a0871c68dc983b9cb0868d4834b2c1`
- `test_serine_provenance_corrected.jsonl`：`56f877bb998701149954b8c01e86b59ecb8503b01742bfd8200e985b564d236b`
- `methylated_new_candidates.csv`（旧 1,333 条排除池）：`6c7b20e96d8b75fa8c09e5d773326b1c38be7bea84e1bf87f86c27d1894d06f3`

三个校正/排除输入默认从仓库的 `v9_inputs/` 读取。运行器会对原始模型、train/test/prior、native、best、historical 和计划文件强制核对冻结 SHA-256；环境变量只允许指向字节完全相同的副本，换数据会按科学合同失败。

若这些冻结文件位于别处，可用环境变量指向其副本：

```bash
V9_TRAIN_JSONL=/path/train_serine_provenance_corrected.jsonl \
V9_TEST_JSONL=/path/test_serine_provenance_corrected.jsonl \
V9_NATIVE_JSONL=/path/17_complexes_native.jsonl \
V9_BEST_CSV=/path/best_designs.csv \
V9_HISTORICAL_CSV=/path/all_designs.csv \
V9_PRIOR_CSV=/path/prior_methylated_candidates.csv \
V9_OUTPUT_ROOT=/path/cyclic_stability_v9_1700 \
bash run_cyclic_stability_v9_1700.sh
```

运行器依次执行：完整单测、从原始 `frankenstein_v28.pt` 重训、151 条内部开发集与 17 个 native control 审计、初始 42,500 次 `T=0.5` 生成、仅对生成期稳定候选缺额靶点自适应续跑、精确 `L×L` cyclic-base 打分、全可行池17×100选择、batch-size-1 重放与打包。精确 base 打分按靶点保存可校验 checkpoint，断点后可继续；科学门失败时不会自动删除或覆盖证据。

## 最终交付判据

只有最终 manifest 的 `quality_gate` 为 `PASS`、全部 `quality_checks` 为真、四个最终文件哈希仍匹配、CSV/FASTA均为1700条且每靶点100条时，才能发送以下文件：

- `final_independent_replay_handoff/1700_详细审计.csv`
- `final_independent_replay_handoff/1700_给尚哥_极简.csv`
- `final_independent_replay_handoff/1700_给尚哥_结构输入.fasta`
- `final_independent_replay_handoff/v9_1700_independent_replay.csv`
- `final_independent_replay_handoff/v9_1700_independent_replay_manifest.json`

极简表只保留结构生成所需的 ID、靶点、带小写甲基化标记的序列、天然序列和甲基化位点；完整概率矩阵、稳定性、base 分数、去重和集中度证据保留在详细审计表与 manifest 中。

## 本工作区的执行状态

本工作区没有 PyTorch/CUDA，也没有已经重训出的 V9 checkpoint，因此这里只能完成根因审计、代码修复、静态回归和 CPU 合成数据测试；不能把尚未在 GPU 上重训和独立回放的候选称为“已通过的 1,700 条”。一键运行器会在缺少 CUDA、输入文件或任一科学门失败时立即停止。
