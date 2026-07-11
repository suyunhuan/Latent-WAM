# AGENTS.md

本文件为在此代码库中工作的编码智能体提供指导。

## 重要规则

除非用户明确要求使用其他语言，否则必须使用中文回复用户。

## 项目概述

Latent WAM 是基于 NAVSIM v2.1.2 的相机自动驾驶研究代码。正式模型位于
`navsim/agents/latent_wam/`；`navsim/agents/transfuser/` 是保留的上游 baseline。
WorldMirror 以独立 git submodule 提供，只用于推理和训练缓存中的几何监督。

英文 `README.md` 与 `docs/latent_wam/` 是权威用户文档，`README_zh.md` 是中文说明。

## 环境

```bash
conda env create -f environment.yml
conda activate latent-wam
pip install -e .
```

正式环境为 Linux、Python 3.10、PyTorch 2.4.1、CUDA 12.4。不要在代码中写入
个人机器的绝对路径；路径通过 Hydra 或环境变量提供。

## 检查与测试

```bash
python -m pytest tests/latent_wam -q
python -m compileall -q navsim/agents/latent_wam scripts
python scripts/check_release_hygiene.py
pre-commit run --all-files
```

CPU CI 覆盖配置、缓存和 checkpoint 映射；GPU smoke test 覆盖正式权重加载与
一次推理。不要求在 CI 中运行完整训练或 NAVSIM 评估。

## Latent WAM 约束

- 公开命名统一使用 `Latent WAM`、`latent_wam`、`LatentWAM*`。
- 推理只读取当前三路图像和当前 ego status。
- 输出键为 `trajectory`、`all_trajectories`、`mode_logits`。
- 训练时间点固定为 `[-1.5, 0.0, 2.0, 4.0]` 秒。
- 只保留 teacher-forcing `forward_train`，不要重新引入 curriculum/refine 分支。
- checkpoint 迁移必须显式列出 rename/drop 规则，并保持严格键验证。
- 旧 gzip-pickle 缓存只能作为可信输入读取；公开格式必须带 schema version。
- 最终实现的 loss 行为是 WTA L1 trajectory、MSE world model、cosine geometry、
  command/velocity/acceleration ego-status loss。

## 代码风格

- 最大行长度 120，使用 Black 与 isort。
- 公共接口提供类型注解和 Google 风格 docstring。
- 导入第三方大型组件时尽量延迟加载，使配置和迁移工具可在 CPU 最小环境运行。
- 保留用户无关改动，不使用破坏性 git 命令。

## 许可证

不要把整个仓库无条件描述为 Apache-2.0。WorldMirror 使用 Tencent
HunyuanWorld-Mirror Community License Agreement，并包含地域、用途和再分发限制。
修改发布说明时必须保留 `NOTICE` 与 `THIRD_PARTY_LICENSES.md` 的披露。

## Agent skills

### Issue tracker

任务与规格以 Markdown 文件形式保存在 `.scratch/`。参见
`docs/agents/issue-tracker.md`。

### Triage labels

使用默认的五种 triage 状态。参见 `docs/agents/triage-labels.md`。

### Domain docs

使用单上下文领域文档布局。参见 `docs/agents/domain.md`。
