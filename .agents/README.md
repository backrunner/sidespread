# Sidespread — Agent 设计文档目录

本目录存放 sidespread 项目的设计文档，供后续开发会话参照。

## 文档索引

| 文档 | 内容 | 何时读 |
|---|---|---|
| [requirements.md](./requirements.md) | 需求文档：背景、输入输出、功能/非功能需求、CLI 接口、验收标准 | 开始任何功能开发前 |
| [architecture.md](./architecture.md) | 架构与模块划分：顶层架构、目录结构、各模块职责、数据流、依赖 | 写代码前了解整体设计 |
| [implementation_plan.md](./implementation_plan.md) | 实现方案：9 个阶段（P0-P8）的交付物、验证标准、风险与缓解、工时 | 制定 sprint / 落地代码前 |
| [research_notes.md](./research_notes.md) | 调研结论：候选模型对比、FlashSR vs UniverSR benchmark、选型理由、文献空白 | 优化算法 / 更换模型 / 写论文时 |
| [real_music_benchmark.md](./real_music_benchmark.md) | 真实音乐的保守路由、全覆盖感知补足与 DSP/神经对照基准 | 调整检测或修复算法前 |

## 一句话项目摘要

修复 AI 生成音乐（Suno 类）side 通道高频缺损的 Rust CLI 工具。用户只需运行一次 `process`；健康段跳过，缺损段统一使用经过真实音乐基准验证的 DSP 延展，输出修复 WAV + 评估报告。UniverSR ONNX 保留为隐藏的研究评测后端。

## 关键决策（不要遗忘）

1. **公开默认路线 = 加法式 DSP**：保留原 Side 复数谱，只叠加缺失能量；校准强度 2.0、60° 相位扩散，处理所有检测为缺损的段。40 首分层 FMA 样本中比 UniverSR 更接近健康 R_hf/ICCC 分布。
2. **UniverSR 只用于研究评测**（arXiv:2510.00771，MIT，音乐训练，229MB，全卷积+iSTFT）。当前 CPU 约 40x RTF，8 类 FMA 对照指标弱于 DSP，不向普通用户暴露模式选择。详见 real_music_benchmark.md。
3. **感知补足，不声称原始还原**：HF-SNR 保留为诊断；默认优化健康能量分布、LSD/MCD、ICCC、受保护频段透明度和听感。
4. **削波安全**：优先完整保留新增 Side，必要时对 M/S 使用同一固定增益；衰减到 3 dB 上限后才缩小合成差分。
5. **检测先行**：不需要处理的音频直接提示用户退出。
6. **实验推理运行时 = `ort` (ONNX Runtime)**，不用 tch-rs。详见 research_notes.md §6。

## 技术栈

Rust 2021 · `hound` · `realfft` · `ndarray` · `ort`(ONNX Runtime, CPU/CUDA/ROCm/CoreML via `ep`) · `rubato` · `clap` · `serde`
