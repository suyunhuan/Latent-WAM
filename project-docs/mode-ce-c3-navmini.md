# Mode CE 增训试跑（C3，勿作发布）

训练侧增加可选 `losses.mode_classification`（默认 **0.0**，发布路径不变）与
`trajectory_lateral_weight`（默认 1.0）。开启 CE 时对 `mode_logits` 监督 WTA
`best_mode` 和/或 GT command。

## navmini-64 短增训结果（负向）

在发布权重上、既有 `latent_wam_cache_navmini`（64 scenes）短增训
（`mode_classification=0.1`、`trajectory_lateral_weight=1.5`、lr `1e-4`、15 epoch）后，
smoke100 ePDMS×100 约 **84**（command ~84.4 / logits ~81.7），相对发布基线 ~89 **明显回退**。

**不要**用该配方/权重替换发布推理。保留代码钩子供更大 cache / 更保守 lr 再试；
推理继续用发布权重 + 可选 C1 `mode_logits`（见 [mode-select-inference-ab.md](mode-select-inference-ab.md)）。
