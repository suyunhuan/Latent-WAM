# AutoDL bootstrap / ePDMS status（公开摘要）

**Updated:** 2026-10-23 · RTX 4090D · 工作区在数据盘 `latent_wam_workspace`  
本文件由 Project store 的 internal 状态**去敏精简**而来：不含 SSH 主机/端口、密码或密钥。

## 给用户

**完整 navtest ePDMS ≈ 89.34；已挑 6 个 case 出官方 BEV/相机可视化。**  
图片见本目录 [`media/epdms-case-viz/`](media/epdms-case-viz/)（每例：`*_bev.png` 评分轨迹、`*_cameras.png` 八路相机+BEV、`*_bev_live.png` 现场重推）。

## Process state

| Step | Status |
| --- | --- |
| Phase A / data / metric cache | **DONE** |
| ePDMS smoke `--max-scenarios 100` | **DONE** (~89.08) |
| ePDMS **full** navtest | **DONE** (~89.34) |
| Case visualization (6 tokens) | **DONE** |

## Full result

| Field | Value |
| --- | --- |
| Rows | 12146 scenarios + `average_all_frames` |
| **ePDMS (×100)** | **89.343…** (`score=0.893431`) |

### `average_all_frames` (selected)

| Metric | Value |
| --- | --- |
| no_at_fault_collisions | 0.98127 |
| drivable_area_compliance | 0.973078 |
| driving_direction_compliance | 0.995719 |
| traffic_light_compliance | 0.99753 |
| ego_progress | 0.877032 |
| time_to_collision_within_bound | 0.973078 |
| lane_keeping | 0.976124 |
| history_comfort | 0.98304 |
| two_frame_extended_comfort | 0.879283 |
| **score** | **0.893431** |

## Smoke (reference)

| Field | Value |
| --- | --- |
| ePDMS ×100 | **89.08** |

## Case visualization (6)

Uses official `navsim.visualization` (`plot_bev_frame` / `plot_cameras_frame` / `plot_bev_with_agent`). Scored trajectories from full-eval outputs. **未改模型/训练代码。**

| Tag | Token | Score | Highlights | Media |
| --- | --- | --- | --- | --- |
| fail_collision_dac | `afbf26b6d3bb5bee` | 0.00 | col=0, dac=0, prog=1 | `fail_collision_dac_afbf26b6d3bb5bee_{bev,cameras,bev_live}.png` |
| fail_collision | `e5e1dbb69578517b` | 0.00 | col=0, dac=1, prog≈0.61 | `fail_collision_e5e1dbb69578517b_{bev,cameras,bev_live}.png` |
| fail_dac | `afd632f043e35a49` | 0.00 | col=1, dac=0, prog≈0.78 | `fail_dac_afd632f043e35a49_{bev,cameras,bev_live}.png` |
| mid_lane_comfort | `0367791336c3533e` | 0.557 | lk=0, comfort=0, prog≈0.78 | `mid_lane_comfort_0367791336c3533e_{bev,cameras,bev_live}.png` |
| mid_score | `b7648a9b27e7574e` | 0.557 | soft fail (TTC/comfort band), prog≈0.98 | `mid_score_b7648a9b27e7574e_{bev,cameras,bev_live}.png` |
| high_perfect | `067d731005885300` | 1.00 | all hard metrics 1.0 | `high_perfect_067d731005885300_{bev,cameras,bev_live}.png` |

- Manifest: [`media/epdms-case-viz/manifest.json`](media/epdms-case-viz/manifest.json)
- Verified: 18 PNGs + manifest present in this export

## Env notes（无密钥）

- Metric cache reused; Ray limit `worker.threads_per_node=16` for cache rebuilds
- DINOv2 via `HF_ENDPOINT=https://hf-mirror.com`
- `huggingface-hub==0.36.2`（配合 transformers 4.45.2）
