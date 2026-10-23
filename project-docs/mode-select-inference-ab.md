# 推理 mode_select A/B（C1 探针）

可选推理轨迹模式选择：默认仍为发布基线 `command`；可用 `--mode-select` /
`LATENT_WAM_MODE_SELECT` 切换到 `mode_logits` 或 `hybrid`。实现见
`navsim/agents/latent_wam/mode_select.py`。

## Smoke 100（navtest 前 100，发布权重）

| Strategy | ePDMS×100 |
| --- | ---: |
| `command`（基线） | 89.08 |
| `mode_logits` | 89.14 |
| Δ | +0.05 |

结论：`mode_logits` 在该探针上略优于 / 至少不劣于 `command`；默认评测路径不变，
实验评测可选用 logits。全量 navtest 与 mode CE 训练对齐见后续工作。

```bash
# 默认：与发布行为一致
python scripts/evaluate_latent_wam_epdms_parallel.py ... 

# 实验：按 mode_logits 选轨迹
python scripts/evaluate_latent_wam_epdms_parallel.py ... --mode-select mode_logits
# 或: LATENT_WAM_MODE_SELECT=mode_logits ...
```
