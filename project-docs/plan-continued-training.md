# 增训 → 诊断 → 改进 → 重训

## 目标

在已复现官方推理（navtest ePDMS ≈ **89.34**）的前提下：

1. **先跑通增训**（从发布权重继续训练，单卡/小规模亦可）
2. **再诊断**当前模型/训练的问题（结合失败 case 与代码）
3. **提出并落地**架构或训练改进（改代码和/或补数据）
4. **重训或增训**后对比 ePDMS

## 阶段

### A. 跑通增训（当前 · 进行中）

- 机器：既有 AutoDL RTX 4090D（SSH 直控）；数据盘 ~550GB，空闲曾约 423GB
- **磁盘事实：** 完整 **navtrain**（current+history×32）压缩约 **~430GB**，当前盘装不下全量。Phase A 改用官方更小可训子集 **`navmini`**（OpenScene **mini 相机** + 官方 `scene_filter/navmini.yaml`）
- 权重：WorldMirror（仅 cache）+ 已有 DINOv2 / `model.safetensors`
- 启动：官方 `run_latent_wam_training_multi_node.sh` **缩成 1×4090**；`agent.checkpoint_path=…/model.safetensors` weights-only（`resume_ckpt_path=null`）；`temporal_attention_mask=block_diagonal` 对齐发布权重
- AutoDL 脚本（均在 `/root/autodl-tmp/`）：
  - `download_train_min.sh`（tmux `lw-train-dl`）
  - `run_cache_navmini.sh`（`MAX_SCENES=64`）
  - `run_continued_train_1gpu.sh`（先 `smoke_train_latent_wam.py` 再 Lightning）
  - 编排：`run_phase_a_train.sh`（tmux `lw-train`）
- 日志：`/root/autodl-tmp/download-train.log`、`/root/autodl-tmp/phase-a-train.log`
- 验收：smoke 一步有限 loss + 至少 1 epoch/若干 step 存 ckpt
- 状态见 [`autodl-bootstrap-status.md`](autodl-bootstrap-status.md)

### B. 问题分析

- 结合 [代码库分析](latent-wam-codebase-analysis.md)、[case viz](media/epdms-case-viz/manifest.json)（碰撞/DAC/车道舒适度失败）
- 输出：瓶颈清单（架构 / loss / 数据 / 评测分布）与优先级

### C. 改进落地

- 改架构或训练配方 / 补数据；保持可对比的实验记录
- 增训或重训 → 与基线 **89.34** 对比（smoke + 必要全量）

## 约束

- Phase A：优先官方训练路径；仅为单卡/磁盘做配置与脚本适配，不先做大改架构
- 中文沟通；状态见 Project Notes；细节进本计划与分析文档

## 关联

- [代码库技术分析](latent-wam-codebase-analysis.md)
- [推理 Runbook](runbook-navisim-inference.md)
- Fork：https://github.com/suyunhuan/Latent-WAM
