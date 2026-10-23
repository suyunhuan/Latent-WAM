# Plan: Latent WAM inference on NAVSIM

## Outcome
Unmodified fork runs **inference** on **NAVSIM** (user said “navisim”; treat as NAVSIM unless README says otherwise). Clear runbook + list of what still needs GPU / data / cloud.

## Steps
1. Inventory fork: README, env, weights, NAVSIM hooks, inference entrypoints.
2. Map prerequisites: Python/CUDA, NAVSIM install, datasets/checkpoints, licenses.
3. Attempt cloud setup as far as possible without code edits; record exact commands.
4. If GPU / large data / NAVSIM runtime is blocked in cloud, produce a minimal handoff for user machine or rented GPU.
5. Smoke-run inference (or document the first failing prerequisite with a fix path).

## Out of scope (later)
- Training / renting cloud GPUs for full train
- Architecture survey and model integration
