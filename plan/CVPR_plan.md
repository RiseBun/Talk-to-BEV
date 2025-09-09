# CVPR 2026 冲刺计划（Talk-to-BEV）

> **当前日期**：2025-09-09 （时区：America/Los_Angeles）  
> **关键节点（请以官网为准）**：
- **ICLR 2026**：摘要 **Sep 19, 2025 (AoE)**；全文 **Sep 24, 2025 (AoE)**  
- **CVPR 2026（OpenReview 显示）**：摘要 **Nov 7, 2025 (UTC)**；全文 **Nov 14, 2025 (UTC)**  
- **ICRA 2026**：论文提交 **Sep 15, 2025 (PST)**；视频窗口 **Aug 5–Sep 9** 与 **Sep 17–22, 2025**

## 目标
- **主目标**：CVPR 2026 主会论文（Talk-to-BEV）
- **次目标**：ICLR 2026（方法短版/语义编译理论与仿真）

## 周计划（到 CVPR 截稿）
**W1（~ Sep 15）**  
- 完成 DSL 扩展（`spatial_masks`/`risk_map`）与 OCP 注入接口落地（距离约束 + 风险代价 + Hybrid 切换）。  
- 跑通 2 个核心场景（Sparse-Static / Long-Corridor），产出首批轨迹可视化。

**W2（Sep 16–Sep 22）**  
- 接入风险通胀（速度/方差）与 Robust-CBF 裕度接口；  
- 场景扩展到 4 类；实现批量评测脚本，产出初版主表。  
- ICLR：整理方法简版 + 仿真曲线，决定是否提交（Sep 19/24）。

**W3（Sep 23–Sep 29）**  
- 上线多 LLM 共识；完成 5 类场景复现实验与消融。  
- 录制 30s demo（语言→BEV→轨迹）。

**W4（Sep 30–Oct 6）**  
- 扩充到 7 类场景，每类≥20 episode；  
- 补充鲁棒性（遮挡/语言噪声）与失败案例库；  
- 完成 60–90s 视频 v1。

**W5（Oct 7–Oct 13）**  
- 全量重复实验（≥3 种子），固化表格与图；  
- 写作：Abstract/Intro/Related/Method 初稿。

**W6（Oct 14–Oct 20）**  
- 写作：Experiments/Discussion/Limitations；整理附录；  
- 代码清理：一键脚本、随机种子、参数表。

**W7（Oct 21–Oct 27）**  
- 内部审稿/改图改文；视频 v2；  
- 预审查：匿名性、伦理与安全声明。

**W8（Oct 28–Nov 3）**  
- 终稿冻结 v1；补充材料定稿（视频、更多场景、参数表、失败案例）。

**W9（Nov 4–Nov 7）**  
- **提交摘要（CVPR）**；打磨正文。

**W10（Nov 8–Nov 14）**  
- **提交全文（CVPR）**；检查上传文件完整性。

## 交付物 Checklist
- [ ] 代码：支持 `spatial_masks`/`risk_map` → OCP（硬/软/Hybrid）
- [ ] 风险通胀 + Robust-CBF 裕度接口
- [ ] 7 类场景批量跑脚本与结果 CSV
- [ ] 主结果大表 + 消融 + 鲁棒曲线 + 实时性箱线图
- [ ] 60–90s 视频（语言→BEV→轨迹三分屏 + 遮挡鲁棒 + 指标条形图）
- [ ] 论文 LaTeX：主表 `main_table.tex` 与模板
- [ ] 双匿名/伦理/复现脚本（seed 固定、参数写入）
