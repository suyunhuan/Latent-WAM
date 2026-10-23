# Latent WAM 项目文档（Project handbooks）

本目录由 Cursor Project 工作区导出，**不属于上游 NAVSIM 官方文档**。内容覆盖 fork 实勘、NAVSIM 推理/ePDMS 复现、GPU 租机与后续增训计划。模型与训练代码本身未因本导出而改动。

Fork：https://github.com/suyunhuan/Latent-WAM

## 文档索引

| 文档 | 说明 |
| --- | --- |
| [project-context.md](project-context.md) | 项目目标、阶段与约束 |
| [latent-wam-codebase-analysis.md](latent-wam-codebase-analysis.md) | 代码库技术分析（结构 / 模型 / 训练 / ePDMS / 实测） |
| [runbook-navisim-inference.md](runbook-navisim-inference.md) | NAVSIM 推理与 ePDMS Runbook |
| [plan-navisim-inference.md](plan-navisim-inference.md) | 推理阶段计划 |
| [plan-continued-training.md](plan-continued-training.md) | 增训 → 诊断 → 改进 → 重训计划 |
| [gpu-rental-brief.md](gpu-rental-brief.md) | GPU 租机选型与接入说明 |
| [gpu-bootstrap.md](gpu-bootstrap.md) | 租机后环境引导命令顺序 |
| [autodl-bootstrap-status.md](autodl-bootstrap-status.md) | AutoDL ePDMS 复现状态摘要（已去敏） |
| [mode-select-inference-ab.md](mode-select-inference-ab.md) | C1 推理 mode_select A/B（smoke 89.14 vs 89.08） |
| [mode-ce-c3-navmini.md](mode-ce-c3-navmini.md) | C3 mode CE 钩子说明；navmini-64 FT 回退至 ~84，勿作发布 |

## 媒体

- [media/epdms-case-viz/](media/epdms-case-viz/) — 6 个 ePDMS case 的 BEV / 相机可视化（含 [manifest.json](media/epdms-case-viz/manifest.json)）

## 说明

- 相对链接均以本目录 `project-docs/` 为根；分析文档中的 case 图指向 `media/epdms-case-viz/`。
- 导出时已去掉 SSH 主机/端口、密码与密钥等敏感信息；Project store 内的 `preferences.md` 与完整 internal 脚本未纳入本目录。
