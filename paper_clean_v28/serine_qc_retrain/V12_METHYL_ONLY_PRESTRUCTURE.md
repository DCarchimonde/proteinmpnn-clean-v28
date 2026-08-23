# V12：真实的 17 × 100 甲基化序列预结构流程

## 任务边界

本流程严格按实际交付顺序执行：

1. 复用已经通过单体指标和循环表示审计的 V11 checkpoint；
2. 每个复合物只放行 100 条严格通过模型甲基化门的序列；
3. 合计 1,700 条交给尚哥生成结构；
4. 尚哥返回结构后，才计算真实的 receptor-aligned complex / cyclic-peptide RMSD；
5. 真实 RMSD 不能在第 2 步之前产生，也不能用旧六个靶点训练的预测分数冒充。

甲基化放行定义保持不变：至少一个可甲基化母体残基在完整循环起点表示网格上的
`representation_min` 经八位小数舍入后仍严格 `> 0.6`，并且阈值分歧为 0。
这是模型预测的甲基化标签，不是实验确认。

## 为什么不再使用 base 门和预结构 RMSD 排序

旧的 cyclic-base 分数是 receptor-conditioned ProteinMPNN 的序列似然诊断，不是
RMSD，也不是甲基化实验结果。用户本轮明确要求“只要甲基化的”，因此它不再是
预结构放行门，也不参与 100 条的排序。

旧六靶点的 476 条结构可以用于方法开发，但不能给尚未生成结构的新序列提供真实
RMSD。V12 完全移除了预结构 RMSD 预测排序。最终表会明确记录
`RMSD = NOT_AVAILABLE_UNTIL_SHANGGE_RETURNS_STRUCTURES`。

## 3ZGC 的处理

V11 的全局单体与循环不变性审计已经通过；3ZGC 没有目标特异的真实甲基化标签，
不能把模型自身的伪标签重新当训练真值。V12 因此不再增加 expert-head 轮数，也不
做无限随机抽样，而是：

1. 用同一个 V11 checkpoint 回放当前、历史、旧结构批次以及可找到的 V8 3ZGC
   序列；
2. 以完整序列的 `representation_min` 为目标，进行固定轮数的单突变与确定性
   多突变 beam 搜索；
3. 排除历史、当前 raw pool、天然序列以及正向循环等价重复；
4. 对候选重新做完整注释；
5. 用 batch size 1 再独立复算；
6. 只有正好 100 条通过才允许进入 17 × 100 选择。

搜索日志中的“scored sequences”只是 GPU 上评估的序列变体，不是生成的 PDB，
也不是要交给尚哥的额外结构任务。最终交付始终是每靶点 100 条。

## AutoDL 安全启动

V12 直接复用当前 V11 输出目录，不会重跑 V11 训练、42,500 条初始生成或 31,200
条补采样：

```bash
bash launch_v12_autodl_safe.sh
```

安全启动器使用独立后台 supervisor。网页或 SSH 断开不会停止 GPU 任务；程序失败
只会写入状态和退出码，不会把交互终端带着退出：

```bash
cat /root/autodl-tmp/v12_launcher.status
tail -n 160 -F /root/autodl-tmp/v12_launcher.log
cat /root/autodl-tmp/v12_launcher.exitcode
```

只有日志出现以下终态且退出码为 0 才能使用交付文件：

```text
===== V12 ALL PRE-STRUCTURE METHYLATION GATES PASSED =====
Rows: 1,700 = 17 targets x exactly 100
```

最终文件位于：

```text
paper_clean_v28_outputs/cyclic_native_v11_1700_monomer/v12_methyl_only/
  final_independent_replay_handoff/
    1700_给尚哥_极简.csv
    1700_给尚哥_结构输入.fasta
    1700_详细审计.csv
    1700_独立逐条甲基化复算.csv
    v12_1700_methyl_only_independent_replay_manifest.json
```

任一靶点少于 100 条或任一 batch-one 回放不一致时，最终交付不会被授权。
