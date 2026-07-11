# Latent WAM

Latent WAM 是一个面向相机自动驾驶的 latent world-action modeling 研究实现。
推理时，模型仅使用当前时刻的左/前/右三路图像和自车状态，预测未来 4 秒的
自车轨迹；训练时，时序 latent world model 和 WorldMirror 几何特征作为辅助监督。

英文 [README](README.md) 是权威文档；本文件提供对应的中文使用说明。

## 发布内容

- 只保留一个正式模型 `LatentWAMModel`，删除 W4D、LAW、RFT 和消融版本。
- 推理接口只返回 `trajectory`、`all_trajectories`、`mode_logits`。
- 保留 epoch-84 实际使用的 teacher-forcing 训练路径，移除未启用的课程学习、
  自回归、轨迹 refine 和 staged loss 分支。
- 提供带版本号的新缓存格式，以及旧 `transfuser_feature.gz` 两级缓存适配器。
- 提供 Lightning checkpoint 到 `safetensors` 的显式转换工具。
- 恢复并保留独立的官方 NAVSIM v2.1.2 TransFuser baseline。

## 安装

正式支持 Linux、Python 3.10、CUDA 12.4 和 NVIDIA GPU。CPU 可用于配置检查、
缓存迁移、权重转换与单元测试，模型推理建议使用 GPU。

```bash
git clone --recurse-submodules <repository-url> latent-wam
cd latent-wam
conda env create -f environment.yml
conda activate latent-wam
pip install -e .
```

完整环境变量和数据集配置见
[英文安装文档](docs/latent_wam/installation.md)。

## 转换原始 checkpoint

```bash
python scripts/convert_latent_wam_checkpoint.py \
  /path/to/epoch84.ckpt \
  artifacts/latent-wam/model.safetensors
```

转换结果保留推理权重，但不保留 optimizer state，因此不能作为“完全一致”的训练
断点恢复。原始 `.ckpt` 仍可由兼容加载器读取。

## NAVSIM 评估

```bash
python navsim/planning/script/run_pdm_score.py \
  agent=latent_wam_agent \
  agent.checkpoint_path=/path/to/model.safetensors \
  train_test_split=navhard_two_stage \
  experiment_name=latent_wam_eval
```

## 缓存与训练

```bash
python navsim/planning/script/run_latent_wam_caching.py \
  agent=latent_wam_agent \
  train_test_split=navtrain \
  cache_path=$NAVSIM_EXP_ROOT/latent_wam_cache
```

```bash
python navsim/planning/script/run_training.py \
  agent=latent_wam_agent \
  agent.build_target_encoder=true \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  cache_path=$NAVSIM_EXP_ROOT/latent_wam_cache \
  train_test_split=navtrain \
  experiment_name=latent_wam_train
```

真实数据的最小完整链路验证见
[中文 smoke test 文档](docs/latent_wam/smoke-test_zh.md)。该流程只使用一个
trainval 样本和一个 test scene，覆盖 WorldMirror 缓存、单步优化、EMA、权重重载
和 PDM 评分。

## 许可证特别说明

NAVSIM 衍生代码使用 Apache-2.0；`worldmirror` 子模块不是 Apache-2.0，而是
Tencent HunyuanWorld-Mirror Community License Agreement。该协议包含地域、用途、
再分发以及“使用输出改进其他 AI 模型”等限制。Latent WAM 权重训练使用了
WorldMirror feature supervision，因此公开权重前需要法律审核。

使用或发布前请阅读 [NOTICE](NOTICE)、
[第三方许可证说明](THIRD_PARTY_LICENSES.md) 和 `worldmirror/License.txt`。
