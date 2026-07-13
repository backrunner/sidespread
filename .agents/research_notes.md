# Sidespread — 调研结论备份

本文档汇总本项目启动前对"AI 生成音乐 side 通道高频缺损补偿"的完整技术调研，作为后续实现与优化的参考依据。

## 1. 问题背景

Suno 类 AI 音乐生成服务产出的立体声音频，在 M/S 编码下：
- `M = (L+R)/2`（mid，含主体内容，高频较完整）
- `S = (L-R)/2`（side，含立体声宽度，**高频缺损**）

目标：检测并补偿 side 高频，使 M/S 频谱与质量对齐。

## 2. 方案双路线

### A 路线（DSP，mid → side 延展）
当 mid 与 side 中频近似时，把 mid 高频能量"借给" side：
- STFT → 高频 bin 幅度折叠 + 相位抖动 → iSTFT。
- 纯 Rust 实现，廉价保真。
- 参考 LavaSR 的 Linkwitz-Riley 交叉分频思路（纯 DSP 频域 mask）。

### B 路线（神经，中频→高频延展）
当 mid 与 side 不近似（side 是独立内容）时，用带宽扩展模型把 side 自身中频向高频外推：
- **定向频段补齐**：保留 side 原中频，只用网络补高频，频段拼接。
- 通过 ONNX Runtime 在 Rust 中推理。

## 3. 检测方案

按帧/分段（80ms，50% 重叠）计算：
1. **高频能量比** `R_hf = E_S(f>f_c) / E_M(f>f_c)`，`f_c ≈ 8 kHz`。
2. **高频谱形状相似度**：LSD / cosine。
3. **高频互相关** `corr_hf(M, S)`。

决策：`R_hf` 正常→skip；缺损 + corr 高→A；缺损 + corr 低→B；中间→混合。全曲正常则提示用户无需处理。

## 4. 评估方案

全部可在 Rust 实现：
- **LSD**（Log-Spectral Distance，高频段）
- **MCD**（Mel-Cepstral Distance，感知）
- **ICCC**（Inter-channel Cross-Correlation，立体声质量）
- **R_hf**（高频能量一致性）
- **对照集**：干净→人为低通 side→修复→对比原始，得 SNR/LSD-ground-truth

## 5. 神经网络模型选型调研

### 5.1 全面筛查的候选（2021–2026）

| 模型 | 年份 | 类型 | 音乐训练? | 权重公开? | 许可 | 备注 |
|---|---|---|---|---|---|---|
| NU-Wave | 2021 | 扩散 | 通用 | 样本为主 | 未明 | 已被超越 |
| NU-Wave 2 | 2022 | 扩散+STFC | 通用 | 难以验证 | 未明 | 常见基线 |
| NVSR | 2022 | DSP+vocoder | 语音 | 是 | MIT | 语音专用 |
| WSRGlow | 2021 | Flow | 通用 | 是 | — | 旧基线 |
| VoiceFixer | 2022 | vocoder | 语音修复 | 是 | MIT | 语音 |
| **AudioSR** | 2023 | 潜在扩散 | **音乐+语音+SFX** | 是 | MIT | 长期基线，已被超越 |
| AudioLDM2 sr_inpaint | 2023 | 潜在扩散 | 通用 | 是 | — | 有 inpainting 模式 |
| AERO | 2023 | GAN+LSTM | **音乐(MUSDB)** | 是 | MIT | 可用但较老 |
| AEROMamba | 2024 | GAN+Mamba | **音乐(MUSDB)** | 是 | CC0 | Mamba ONNX 难 |
| FiPA-SR | 2026 | GAN+FiLM+Mamba | **音乐** | 否 | — | 声称超 AudioSR 但无公开 |
| Wave-U-Mamba | 2024 | Mamba | 语音 | 难以验证 | — | 语音 |
| CTFT-Net | 2025 | Complex-TF | 语音 | 是 | — | 语音 |
| A2SB | 2025 | Schrödinger Bridge | **音乐 44.1k** | 否(NVIDIA 内部) | CC-BY 论文 | 有 inpainting 但不可用 |
| **FlashSR** | 2025 | 蒸馏扩散(1 NFE) | 通用(含音乐) | 是 | **无 LICENSE** | 快但许可阻塞 |
| SAGA-SR | 2025 | DiT+flow | 通用 | 是 | CC-BY | spectral-roll 引导 |
| **AudioLBM** | NeurIPS 2025 | Latent Bridge | 通用(含音乐) | demo only | — | frequency-aware，代码待发 |
| **UniverSR** | ICASSP 2026 | flow matching+iSTFT | **音乐+语音+SFX** | 是 | **MIT** | **本项目选用** |
| Inference-time Scaling | 2025 | 测试时搜索 | 继承 AudioSR | — | — | 提升AudioSR 47% |
| LatentFlowSR | 2026 | flow | 通用 | 待发 | — | |
| FastWave | 2026 | 扩散优化 | 通用 | 是 | CC-BY | 1.3M 参数 |
| LavaSR | 2026 | Vocos+LR交叉 | **语音(音乐未发布)** | 是 | Apache | 架构好但音乐 OOD |
| InspireMusic | 2025 | AR+flow | 音乐 | 是 | Apache | 生成+SR 耦合，不可独立 |
| Bralios WASPAA'25 | 2025 | latent L1+adv | BWE+upmix | 待发 | — | 唯一立体声+BWE 先例 |

### 5.2 关键结论

**AudioSR 已不再是 SOTA**（2025–2026 至少 5 个工作在音乐/通用上超过它），但仍是最多用、最易用的开源基线。

**立即可用且音乐训练的候选**（严格筛选后）：
- **UniverSR** — MIT，229MB，音乐训练（460h 含 MUSDB/MedleyDB/MoisesDB/MAESTRO），vocoder-free iSTFT，全卷积 ConvNeXt-V2 backbone。
- **AERO** — MIT，音乐 MUSDB，GAN+LSTM，较老但可作备选。
- **AudioSR** — MIT，慢（0.6× realtime），可作对照基线。

**不可用的**：
- FlashSR — 无 LICENSE，分发即侵权；3.2GB；ONNX greenfield。
- AudioLBM — 代码未发布。
- A2SB — NVIDIA 内部，无公开权重。
- FiPA-SR — 无公开代码/权重。
- LavaSR — 语音训练，音乐 OOD。
- AEROMamba — Mamba SSM ONNX 难导。

### 5.3 FlashSR vs UniverSR 详细对比（决定性）

来自 UniverSR 论文 Table 1（唯一同口径对比，测试集 FMA-small 音乐 100 样本）：

| 输入 | 指标 | FlashSR | UniverSR | 差距 |
|---|---|---|---|---|
| 8k→48k | 音乐 2f-model(↑) | 18.01 | **23.52** | +5.5 |
| 12k→48k | 音乐 2f-model | 20.46 | **27.99** | +7.5 |
| 16k→48k | 音乐 2f-model | 24.71 | **30.19** | +5.5 |
| 24k→48k | 音乐 2f-model | 27.36 | **33.58** | +6.2 |
| 8k→48k | 音乐 LSD-HF(↓) | 1.31 | **0.98** | −25% |
| 24k→48k | 音乐 LSD-HF | 1.62 | **0.96** | −40% |

**UniverSR 在每个音乐 cell 全面胜出**，差距明显。MOS 测试音乐/语音/SFX 均最高。

| 维度 | FlashSR | UniverSR |
|---|---|---|
| 速度(GPU 5.12s) | ~0.36s (1 NFE) | 较慢(4步ODE) |
| 输入带宽 | 4–32 kHz | 8/12/16/24 kHz |
| 模型大小 | 3.2 GB | **229 MB** |
| 参数量 | 258M+VAE+Vocoder | **57M** |
| ONNX 易度 | 难(LDM+VAE+BigVGAN) | **易(ConvNeXt+iSTFT)** |
| Vocoder | BigVGAN-v2 | **无(iSTFT)** |
| 许可 | **无(All Rights Reserved)** | **MIT** |

### 5.4 选定：UniverSR

理由：
1. **音乐质量明显更好**（核心维度，非毫厘之差）
2. **MIT 许可**，开源分发零障碍（FlashSR 无 LICENSE，即便非商用开源分发也触发版权，包社区会拒收）
3. **229MB + 全卷积 + 无 vocoder + iSTFT 直出**，ONNX 导出和 Rust 集成都容易
4. Suno 输出是 44.1k/48k，side 高频缺损不是整体低带宽，UniverSR 8k 下限足够

## 6. FlashSR ONNX 导出可行性（已核实，留作备选参考）

FlashSR 技术上可导 ONNX：
- 三个子模型（VAE 1.6GB / AudioSRUnet 986MB / SRVocoder 599MB）分别可导。
- `AudioSRUnet.py:349` 硬编码 `checkpoint(...,True)` 必须 monkeypatch 掉。
- `TimestepEmbedSequential` isinstance dispatch 需 `dynamo=True` 导出。
- **1 步 DPM-Scheduler 可在 Rust 用 3 行绕过**：
  ```rust
  let alpha_t = 1.0 / (1.0 + sigma0 * sigma0).sqrt();
  let sigma_t = sigma0 * alpha_t;
  let latent_out = alpha_t * latent_in - sigma_t * unet_output;
  ```
- 工时 ~8–14 人日，但许可阻塞使其成为备选。

## 6. Rust 推理运行时选型（tch-rs vs ort vs 其他）

### 6.1 候选全景（2026-07 核实）

| 方案 | 直接加载 PyTorch? | 需重实现? | 体积 | UniverSR 可行性 | 许可 | 成熟度 |
|---|---|---|---|---|---|---|
| **tch-rs + TorchScript** | 是(.ptc) | 否 | ~1.7GB libtorch | ★★★★★ 技术最易 | MIT/Apache | 高 |
| tch-rs + safetensors | 仅权重 | 是(module tree) | ~1.7GB | ★★★★ | MIT/Apache | 高 |
| **candle** (HF 纯 Rust) | 仅权重 | 是，**但 ConvNeXt-V2 已在树内** | 几 MB 纯 Rust | ★★★★ | MIT/Apache | 高 |
| burn | 仅权重 | 是，无 ConvNeXt-V2 参考 | 几 MB 或 libtorch | ★★★ | MIT/Apache | 高 v0.21 |
| **tract** (Sonos 纯 Rust) | ONNX/NNEF 图导入 | 否 | 几 MB 纯 Rust | ★★★★ | MIT/Apache | 高 v0.23.3 |
| AOTInductor + FFI | 是(torch.export) | 否 | 几十 MB .so | ★★★ FFI 早期 | BSD | 中 |
| ExecuTorch + FFI | 是(.pte) | 否 | ~50KB | ★★ 边缘优先无 Rust | BSD | 高但错位 |
| Python sidecar | n/a | 否 | ~2GB torch | ★★★★★ 零工 | n/a | 高 |
| MLX | 否 | 是 | 仅 Apple | ★ | MIT | 高但 Apple-only |

### 6.2 tch-rs 硬件 backend 核查（决定性）

| Backend | tch-rs | ort (ONNX Runtime) | 判定 |
|---|---|---|---|
| **CPU** | ✅ 默认 ~1.7GB | ✅ 默认 ~50MB | 都行，ort 小 30× |
| **CUDA** | ✅ 全 op（包 libtorch），`Device::Cuda(0)` | ✅ `ep::CUDA` 官方打包 | 都行 |
| **ROCm** | ⚠️ **stock 0.24.0 静默回退 CPU**（[#1015](https://github.com/LaurentMazare/tch-rs/issues/1015) open 2026-06 无 fix），需手改二进制 crate `build.rs` 加 `-ltorch_hip -lc10_hip` + 手装 ROCm libtorch + `LIBTORCH_BYPASS_VERSION_CHECK=1`；无预编译包 | ✅ `ep::ROCm` 官方打包开箱即用 | **ort 胜** |
| **MPS (Apple)** | ⚠️ `Device::Mps` 类型在但：[#777](https://github.com/LaurentMazare/tch-rs/issues/777) 检测失败、[#687](https://github.com/LaurentMazare/tch-rs/issues/687) `onehot`/`scatter_value` panic "Placeholder storage has not been allocated on MPS device"、[#773](https://github.com/LaurentMazare/tch-rs/issues/773) 打印 MPS tensor panic——多个 open 未修 | ✅ `ep::CoreML` 官方打包，Apple Silicon + Intel Mac | **ort 胜** |

### 6.3 选定：`ort` (ONNX Runtime)

**不选 tch-rs**，理由：
1. **ROCm 在 stock tch-rs 上是坏的**——链接器 `--as-needed` 丢 `libtorch_hip`，GPU 静默变 CPU，无上游修复。
2. **MPS 有多个未修 panic bug**——常见 op 直接崩，非生产级。
3. **分发体积 1.7GB vs 50MB**，差 30×。
4. **ort 的 `ep` 模块统一覆盖 CPU/CUDA/ROCm/CoreML/DirectML**，单一代码路径，全是官方打包测试的二进制。
5. **UniverSR 全卷积 + iSTFT**，ONNX 干净导出，不需要 tch-rs 的 autograd/原生 op 优势。tch-rs 的唯一真实优势（训练 / ONNX 导不出的 op）本项目用不上。

**备选保留**：
- **candle**（HF 纯 Rust）——若未来想消灭 C++ 依赖做极小分发，ConvNeXt-V2 已在 candle 树内，ODE+STFT 在 Rust 写。最长期原生 Rust 方案，但需移植 forward。
- **tract**（Sonos 纯 Rust）——若想纯 Rust 无 onnxruntime C++，可导 NNEF 走 tract。Sonos 生产级。
- **Python sidecar**——首里程碑或作 native 端口的数值参考 oracle。

## 7. 文献空白（本项目的机会）

1. **无公开工作专门做 side 通道高频修复**——所有模型把立体声当两个独立 mono。
2. **无模型在 AI 生成音乐退化模式上训练**——AudioSR 自承认对 MP3 式截止孔洞失效。
3. **定向频段修复欠发达**——只有 A2SB（不可用）/ AudioLBM（未发布）接近。
4. **Silaev et al. 2026 证明**：所有现有 SR 模型产出在 embedding 空间可近乎完美区分真假——修复痕迹的硬上限。

## 8. 关键引用

- **UniverSR** — Choi et al., ICASSP 2026. arXiv:2510.00771. github.com/woongzip1/UniverSR · huggingface.co/woongzip1/universr-audio · MIT
- AudioSR — Liu et al., ICASSP 2024. arXiv:2309.07314 · MIT
- FlashSR — Im & Nam, 2025. arXiv:2501.10807 · 无 LICENSE
- AudioLBM — Li et al., NeurIPS 2025. arXiv:2509.17609
- A2SB — Kong et al. (NVIDIA), 2025. arXiv:2501.11311
- NU-Wave 2 — Han & Lee, 2022. arXiv:2206.08545
- NVSR — Liu et al., 2022. arXiv:2203.14941
- AEROMamba — Abreu & Biscainho, 2024. arXiv:2411.07364
- LavaSR — Sharma, 2026. arXiv:2603.07285
- Bralios WASPAA'25 — arXiv:2506.00681（唯一 BWE+stereo upmix 先例）
- Silaev et al. 2026 — arXiv:2601.03443（SR 可分性分析）
- BWE/SR 综述 2026 — arXiv:2605.16681