# 端到端 smoke test

英文 [smoke-test.md](smoke-test.md) 是权威文档。本流程以最低计算成本验证发布链路，
只使用一个 trainval 样本和一个 test scene，不能用 test 数据进行训练。

验收范围包括：真实三视角四帧数据读取、WorldMirror geometry cache 生成与重载、
一次 forward/backward/optimizer/EMA、推理权重保存与严格重载，以及指定 test token
的 metric cache 和单场景 PDM 评分。

2026 年 7 月 11 日验证使用：

- 数据根目录：`/mnt/oss_data/navsim/Egonav/openscene-v1.1`
- trainval token：`a6a284584b19599f`
- test token：`8051c60f20705909`
- test log：`2021.05.25.17.54.41_veh-35_04111_04288`

真实缓存的 geometry shape 为 `[4, 3, 519, 2048]`，四项 loss、梯度和重载推理
均为有限值，EMA target encoder 确实更新。指定场景评分有效，PDM score 为
`0.9180014612343158`，全部乘法合规指标为 `1.0`。

该分数还需要与同一 token、同一 evaluator 版本的历史结果核对；程序成功退出不等于
迁移结果数值一致。单场景 metric cache 建议写入 `/tmp`，最终只保留较小的 CSV/JSON
报告。

阿里云 PPU 环境可通过 `bash scripts/setup_ppu_smoke_env.sh` 重建项目 overlay；
它继承宿主机与硬件匹配的 torch/torchvision，其余 smoke 依赖固定在
`requirements-ppu-smoke.txt`。
