# Latent WAM — Project Context

## Goal
Reproduce and extend **Latent WAM: Latent World Action Modeling for End-to-End Autonomous Driving**, then survey related latent world-action architectures and land an integrable, trainable baseline.

## Repo
- User fork: https://github.com/suyunhuan/Latent-WAM
- Policy for phase 1: **do not change model/training code** — only run inference as published.

## Phases
1. **Inference on NAVSIM** — install deps, obtain weights, run official/inference path on NAVSIM; document exact commands and blockers.
2. **Cloud training** — rent GPU cloud; train (or fine-tune) with published recipe once inference is proven.
3. **Architecture survey + integration** — collect typical AD latent-WAM designs; propose a coherent merge; implement and get training running.

## Constraints
- Phase 1: no code changes for “making it train”; setup/scripts/docs only if strictly required to run inference.
- Prefer official README / configs over reinventing the pipeline.
- Chinese communication with the user; keep durable docs clear and linkable.

## Decisions
- (none yet beyond phased plan above)

## Docs
- [代码库技术分析](latent-wam-codebase-analysis.md) — 仓库结构、模型/训练/ePDMS、实测与阅读地图
