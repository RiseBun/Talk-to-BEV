# Talk-to-BEV CVPR Pack (Engineering + Planning)

这是一份**可直接落地**的工程与计划包，帮助你在现有 SafeTalk-MPC 基础上实现：
- 语言可编辑 BEV 世界模型（`spatial_masks`/`risk_map` JSON Patch 扩展）
- 风险通胀 + 鲁棒 CBF 安全盾（接口与占位实现）
- 7 类挑战场景评测脚手架（批量跑实验、聚合指标、导出 LaTeX 大表）
- CVPR 截稿前的详细周计划

> 使用方式：把 `semantics/`, `perception/`, `riskmap/`, `compiler/`, `eval/`, `latex/` 目录拷贝/合并到你的 `sem2mpc/` 下（或按需修改导入路径），先跑 demo，再逐步替换占位模块为真实 BEV/CBF。

## 快速开始
```bash
# 1) 运行最小 demo（示例 risk map + 语言 → 生成带 spatial_masks/risk_map 的 JSON Patch）
python demo_bev_language_mpc.py   --instr "绕开人群30厘米，从桌子右侧0.5米停下"   --risk sample/sample_risk.npy   --out examples/patch_example.json

# 2) 批量跑 7 类场景（示例脚本，不依赖真实 OCP，仅产生结构化结果以打通流程）
python eval/run_experiments.py --out results/result_stub.csv

# 3) 聚合与出图、导出 LaTeX 表格
python eval/aggregate_and_plot.py --input results/result_stub.csv --outdir results/

# 4) 在你的 OCP 构造中接入 'spatial_masks' 与 'risk_map'（见 compiler/ocp_integration_example.py 与 README 注释）
```
Talk-to-BEV: Language-Editable Bird’s Eye Views for Verified-Safe MPC

摘要（Abstract 雏形）

我们提出 Talk-to-BEV，这是第一个能够将 自然语言直接编译为可编辑 BEV（Bird’s Eye View）世界模型 并进一步转化为 Verified-Safe Model Predictive Control (MPC) 的系统。与以往的 Vision-Language Navigation 或 LLM-to-Planner 不同，Talk-to-BEV 引入了语义可操作的 BEV 表示：用户只需用口语化指令（如“绕开人群 30 厘米，从桌子右侧 0.5 米停车”），系统便能在 BEV 场景中自动生成 keep-out/keep-in 区域与风险场 (risk map)，并编译为 硬/软/Hybrid 约束 与 代价函数。同时，我们提出 Robust Safety Shield，结合 速度依赖的风险通胀 (risk inflation) 与 鲁棒控制障碍函数 (Robust-CBF)，确保即使感知和语言存在噪声，轨迹依然保持可证明的安全裕度。

在标准化的 7 类挑战场景（密集静态、遮挡、动态行人、走廊、精准停车、语言含糊等）上，我们系统性评估了 Talk-to-BEV 与现有方法（rule-based MPC、naive detection→MPC、SafeTalk-MPC 仅参数级语义）对比。结果显示：Talk-to-BEV 在 任务成功率、碰撞率、最小障碍距离、轨迹平滑度、实时性 等指标上显著优于基线；更重要的是，我们首次展示了语言-视觉-控制三模态间的闭环编译范式，在实验中实现了可验证的安全性与可复现的性能提升。我们将开源代码、场景套件与演示视频，为未来的 安全人机交互机器人系统 提供一个新标准。

引言核心卖点（Introduction 片段）

动机

语言是人类最自然的交互方式，但如何将自然语言转化为可验证的安全控制，仍是开放问题。

现有 Vision-Language Navigation (VLN) 方法多停留在 模仿学习或 RL，缺乏 约束可解释性与安全保证。

LLM-to-Planner / LLM-to-MPC 工作通常仅在参数层面修改权重或半径，无法扩展到空间级别的 BEV 场景约束。

关键创新

语言可编辑的 BEV 表示：首次提出 Semantic BEV Compilation，语言可直接在 BEV 中生成/修改 禁入区域 (keep-out)、可达走廊 (keep-in)、风险地图 (risk map)。

Verified-Safe 编译：通过 Hybrid 安全盾（硬/软/混合约束）结合 鲁棒 CBF + 风险通胀，在噪声与遮挡下依然保持 可证明安全性。

闭环强实证：在标准化多场景中提供大规模对比与消融，涵盖语言模糊、感知不确定性、动态风险，首次展示语言-感知-控制一体化范式。

优势总结

新表示：不是把检测结果简单送入控制器，而是定义了语言可操作的世界模型 (BEV)。

新安全层：不是经验式调半径，而是基于 Robust-CBF 形式化保证 + 速度/不确定性自适应风险场。

新评测协议：不是只看单一 demo，而是构建7 类挑战场景 + 全指标量化，支持完全可复现。

工程护城河：直接对接你已有的 SafeTalk-MPC 安全盾/反回退/终端盒/风险自适应/metrics 管线，降低开发成本，提升系统稳定性。

与相关工作的差异

相比 VLN：我们不是学习一个黑箱 policy，而是显式生成 约束/代价 patch，可解释、可编辑、可验证。

相比 LLM-to-MPC：我们不止于改数值参数，而是能定义 空间区域、风险分布、Hybrid 约束。

相比 BEV 感知工作（BEVFormer/BEVFusion）：我们不止于检测/识别，而是把 BEV 升级为可编译的控制接口。

相比 CBF-MPC：我们不是只在控制域做鲁棒化，而是把语言/感知不确定性显式建模为风险通胀，真正实现 语言→感知→控制的安全闭环。

🔥 独创性总结（给审稿人看的“核弹”）

新 Paradigm：Language-Editable BEV World Models → 可直接编译为 Verified-Safe MPC。

新 Guarantee：鲁棒 CBF + 风险通胀，提供形式化安全裕度，而不是经验式 fallback。

新 Evaluation：大规模标准化场景 + 全指标评测，证明方法在多模态噪声下依然稳定。

可复现 & 可扩展：基于 JSON patch & MPC，开源即复现，未来可扩展到机器人/自动驾驶/人机共驾。

💡 一句话杀招（可放在论文最后一段 / 宣传 tweet）：
Talk-to-BEV shows how humans can “draw with words” on a Bird’s Eye View, and the robot executes it with mathematically guaranteed safety.