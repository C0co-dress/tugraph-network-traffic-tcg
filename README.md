# 安全通论实验四：TCG图网络流量分类

## 概述
基于TuGraph的Traffic Causality Graph (TCG)网络流量分类实验。将网络流建模为Flow顶点，按照刘珍论文定义四种流间因果关系边（CR/PR/DHR/SHR），使用Node2Vec进行图嵌入，对比Raw vs TCG-only vs Raw+TCG三组特征，评估DecisionTree/KNN/RandomForest/MLP四种分类器。

## 关键结果
- 最佳模型：RandomForest(Raw+TCG)，Weighted F1 = 0.7736
- TCG贡献：对树模型 +5.04pp（Raw 0.7232 → Raw+TCG 0.7736）
- TCG单独使用效果差（F1 0.28-0.50），需与原始统计特征融合

## 复现
```bash
python scripts/run_tcg_only.py --scan-rows 600000 --top-classes 10 --samples-per-class 5000 --embedding-dim 16 --epochs 50
tensorboard --logdir runs/tcg_only
```

## 数据
数据集：IP Network Traffic Flows Labeled with 75 Apps (Kaggle)，需下载到 Dataset-Unicauca-Version2-87Atts.csv/ 目录。

## 文件结构
- scripts/run_tcg_only.py — 实验主脚本
- scripts/render_tcg_report_figures.py — 报告图表生成
- outputs/tcg_only/ — 全部实验结果
- runs/tcg_only/ — TensorBoard日志
- reports/ — 实验报告（DOCX + MD）

