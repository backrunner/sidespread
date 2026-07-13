# Sidespread — Agent 设计文档目录

本目录存放 sidespread 项目的设计文档，供后续开发会话参照。

## 文档索引

| 文档 | 内容 | 何时读 |
|---|---|---|
| [requirements.md](./requirements.md) | 需求文档：背景、输入输出、功能/非功能需求、CLI 接口、验收标准 | 开始任何功能开发前 |
| [architecture.md](./architecture.md) | 架构与模块划分：顶层架构、目录结构、各模块职责、数据流、依赖 | 写代码前了解整体设计 |
| [implementation_plan.md](./implementation_plan.md) | 实现方案：9 个阶段（P0-P8）的交付物、验证标准、风险与缓解、工时 | 制定 sprint / 落地代码前 |
| [research_notes.md](./research_notes.md) | 调研结论：候选模型对比、FlashSR vs UniverSR benchmark、选型理由、文献空白 | 优化算法 / 更换模型 / 写论文时 |

## 一句话项目摘要

修复 AI 生成音乐（Suno 类）side 通道高频缺损的 Rust CLI 工具。双路线：DSP（mid→side 延展，纯 Rust）+ 神经网络（UniverSR ONNX，中频→高频延展）。检测先行，按段智能切换路线，输出修复 WAV + 评估报告。

## 关键决策（不要遗忘）

1. **B 路线模型 = UniverSR**（arXiv:2510.00771，MIT，音乐训练，229MB，全卷积+iSTFT）。FlashSR 因无 LICENSE 被排除（即便非商用开源也分发侵权）。详见 research_notes.md §5。
2. **定向频段补齐**：保留 side 原中频，只用网络补高频，频段拼接，不做整段超分。
3. **ODE 循环在 Rust 控制**，不烘焙进 ONNX，单图导出更稳。
4. **A 路线纯 Rust**：mid/S 相关时用，廉价保真兜底。
5. **检测先行**：不需要处理的音频直接提示用户退出。
6. **推理运行时 = `ort` (ONNX Runtime)**，不用 tch-rs。tch-rs 在 ROCm（#1015 静默回退 CPU，无 fix）和 Apple MPS（#687/#773/#777 多个 open panic）上是硬伤；ort 的 `ep` 模块官方覆盖 CPU/CUDA/ROCm/CoreML/DirectML 且体积 50MB vs libtorch 1.7GB。UniverSR 全卷积 ONNX 友好，不需要 tch-rs 的 autograd/原生 op 优势。详见 research_notes.md §6。

## 技术栈

Rust 2021 · `hound` · `realfft` · `ndarray` · `ort`(ONNX Runtime, CPU/CUDA/ROCm/CoreML via `ep`) · `rubato` · `clap` · `serde`