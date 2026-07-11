# 端到端 smoke test

英文 [smoke-test.md](smoke-test.md) 是权威文档。本流程以最低计算成本验证发布链路，
只使用一个 trainval 样本和一个 test scene，不能用 test 数据进行训练。

验收范围包括：真实三视角四帧数据读取、带相机内外参条件的 WorldMirror
geometry v2 cache 生成与重载、
一次 forward/backward/optimizer/EMA、推理权重保存与严格重载，以及指定 test token
的 metric cache 和单场景 PDM 评分。

2026 年 7 月 11 日验证使用：

- 数据根目录：`/mnt/oss_data/navsim/Egonav/openscene-v1.1`
- trainval token：`a6a284584b19599f`
- test token：`8051c60f20705909`
- test log：`2021.05.25.17.54.41_veh-35_04111_04288`

真实缓存的 geometry shape 为 `[4, 3, 519, 2048]`；与旧 geometry 的平均余弦
相似度为 `0.9998027682304382`，通过 `0.999` 验收阈值。四项 loss、梯度和重载推理
均为有限值，EMA target encoder 确实更新。指定场景评分有效，PDM score 为
`0.9180014612343158`，全部乘法合规指标为 `1.0`。

该分数还需要与同一 token、同一 evaluator 版本的历史结果核对；程序成功退出不等于
迁移结果数值一致。单场景 metric cache 建议写入 `/tmp`，最终只保留较小的 CSV/JSON
报告。

分布式协调已使用本机两个 Gloo 进程验证，覆盖 process group 生命周期、token
一致性、均衡分片（包括空 rank）、汇总和 manifest。真实提取也已使用两张 PPU
和默认 NCCL 后端验证：两个 rank 各生成一个 WorldMirror 缓存；第二次运行在模型
路径故意无效时仍正确跳过两个有效样本，证明没有加载模型。当前仍未执行跨节点
WorldMirror 提取。

进一步的小型复现实验选取了 8 个 trainval log、每个 log 一个场景。两张 PPU
各生成 4 个真实缓存，8 个缓存对全部通过 generation、dtype、shape 和有限值校验；
续跑时在模型路径故意无效的情况下正确跳过全部 8 个样本。随后使用正式
`run_training.py` 完成两卡 Lightning DDP 的 4 个训练 step 和 2 个验证 batch，
所有 loss 均有限。详细结果见
[`repro-8logs-result.json`](repro-8logs-result.json)。该实验验证工程链路，不代表
统计意义上的精度复现。该实验最初发现各 rank 会独立解析 Hydra 时间戳；训练入口
现已在启动子进程前固定父进程 UID。双 PPU 回归只生成一个实验目录，由 rank 0
写 `run_training.log`，rank 1 独立写 `rank_1.log`。

阿里云 PPU 环境可通过 `bash scripts/setup_ppu_smoke_env.sh` 重建项目 overlay；
它继承宿主机与硬件匹配的 torch/torchvision，其余 smoke 依赖固定在
`requirements-ppu-smoke.txt`。
