# Sidespread — 需求文档

## 1. 背景与问题

类似 Suno 的 AI 音乐生成服务在产出立体声音频时，**side 通道（L-R 差信号）在高频段（典型 >8 kHz）存在信息缺损**：mid 通道（L+R 和信号）承载了主体内容、高频较完整，而 side 通道的高频立体声宽度细节缺失，听起来"窄"或"模糊"。

本工具的目标是：**检测这种高频缺损并补偿，使处理后的能量、频谱和空间相关性回到真实音乐的合理分布**，提升 AI 生成音乐的立体声高频表现。缺失的相位与内容不可被精确恢复，因此产品承诺是感知型高频补足，而不是还原原始丢失波形。

## 2. 输入 / 输出

### 输入
- 一个立体声 WAV 文件（44.1 kHz 或 48 kHz，16/24/32-bit PCM 或 float）。
- CLI 参数：阈值、模式、报告路径等（见 §5）。

### 输出
- 修复后的立体声 WAV 文件（与输入同采样率、同位深、同声道数）。
- 一份 JSON 报告：每段的检测结果、所选路线、处理前后评估指标对比。

## 3. 功能需求

### F1 — Mid/Side 解码与重建
- 立体声 WAV → `M = (L+R)/2`，`S = (L-R)/2`。
- 处理完成后 `L = M+S`，`R = M-S`，限幅（soft clip / tanh），输出 WAV。
- 单声道输入应直接报错退出（无 side 可修复）。

### F2 — 缺损检测与路线决策
按帧/分段（如 50–100 ms 段，相邻 50% 重叠）计算：
1. **高频能量比** `R_hf = E_S(f>f_c) / E_M(f>f_c)`，`f_c` 默认 8 kHz（可配）。同时计算截止频率以下完整频段的 `R_intact`；`R_hf < min(0.3, 0.18 * R_intact)` → 缺损。相对判定避免把本来就很窄或近似 mono 的素材强行扩宽。
2. **高频谱形状相似度**：M 与 S 在 `[f_c, Nyquist]` 对数功率谱归一化后的谱距离（cosine / LSD）。
3. **高频互相关** `corr_hf(M, S)`：判定 mid 与 side 是否近似。

**默认决策**（每段独立判定）：

| 条件 | 路线 | 说明 |
|---|---|---|
| `R_hf` 正常（≥ 阈值） | **skip** | 不需要处理 |
| `R_hf` 缺损 | **DSP** | 用 mid 高频和 side 截止频率以下的能量包络补足，并做相位扩散 |

**整体判定**：若全曲 `R_hf` 均正常 → 提示用户"该音频不需要处理"并退出（仅输出检测报告）。

### F3 — A 路线：DSP 补偿（mid → side 延展）
对检测为缺损的段，把 mid 的高频能量"借给" side，同时保护仍然存在的 Side 信息：
- STFT（n_fft=4096/8192，hop=1024，Hann 窗）。
- 对 M 高频 bin 取幅度，按"side 中频能量 / mid 中频能量"比例得到目标幅度；原 S 复数 bin 不替换，只在目标能量更高时叠加差额。
- 相位：以 M 相位为基础做 60° 平滑扩散；若新增向量会抵消原 S，则翻转 180°，确保已有信息不被相消。
- 合成差额强度 2.0 用于补偿短段重叠拼接的能量损失；最终峰值安全会限制实际新增量。
- 交叉淡入（cutoff 附近过渡带平滑混合）。
- 纯 Rust 实现，无外部依赖。

### F4 — 实验评测后端：神经网络补偿（中频→高频延展）
保留 **UniverSR**（arXiv:2510.00771，MIT，音乐训练，229MB，vocoder-free flow matching，iSTFT 直出）用于开发评测，不作为公开 `process` 的默认路径。FMA 对照显示它在当前模型条件下比 DSP 更慢且更容易产生过量 Side 高频：
- **定向频段补齐策略**：把 side 现有中频以下作为 UniverSR 的输入，得到全频段输出，但**只取高频段**，与原 side 中频拼接（交叉淡入避免接缝）。保留 side 中频原始内容，只让网络"补"高频。
- UniverSR 通过 ONNX Runtime 在 Rust 中推理（`ort` crate）。模型导出为 ONNX，前端 STFT/mel 和后端 iSTFT 在 Rust 中复刻。
- 4 步 midpoint ODE solver，循环在 Rust 里控制。
- stereo 输入按通道处理（side 本身就是单通道，天然契合）。

### F5 — 评估与报告
对处理前后的 side 通道计算以下指标，全部可在 Rust 实现：
1. **高频能量一致性（HFC）**：`R_hf` 与 M 高频谱的 L1/L2 距离，处理后趋近 1。
2. **Log-Spectral Distance (LSD)**：高频段 M/S 之间的 LSD，处理前后对比，越小越好。
3. **Mel-Cepstral Distance (MCD)**：mel 倒谱域距离，更贴合人耳感知。
4. **Inter-channel Cross-Correlation (ICCC)**：高频段 L/R 互相关，应在合理区间。
5. **处理前后对照集**：提供 `sidespread eval` 子命令，对"干净立体声 → 人为低通 side → 修复 → 对比原始"流程，得到带 ground truth 的 SNR/LSD-ground-truth。

报告输出为 `report.json` + 终端摘要表格。

## 4. 非功能需求

### N1 — 技术栈
- **主语言：Rust**（2021 edition）。
- WAV I/O：`hound`。
- FFT：`realfft` / `rustfft`。
- 数值：`ndarray`。
- 神经推理：`ort`（ONNX Runtime Rust binding）。
- CLI：`clap`。
- 重采样：`rubato`。
- 不依赖 Python / PyTorch 运行时（UniverSR 通过 ONNX 在 Rust 内推理）。
- 推理运行时 = `ort`（ONNX Runtime），不用 tch-rs（tch-rs 在 ROCm 静默回退 CPU #1015、MPS 多个 panic bug #687/#773/#777，且 libtorch 1.7GB vs ort 50MB）。ort 的 `ep` 模块统一覆盖 CPU/CUDA/ROCm/CoreML/DirectML。

### N2 — 性能
- A 路线（DSP）：实时处理（RTF < 0.1）。
- B 路线（神经）：不强制实时，但应可在消费级 CPU 上完成（57M 参数 + 4 步 ODE，预期可行；落地前实测）。
- 内存峰值 < 2 GB。

### N3 — 可分发
- 默认处理只需单一 Rust 二进制；实验神经评测可另行下载 ONNX 模型（229 MB）。
- 不捆绑 PyTorch / CUDA。
- 本项目代码采用 Apache-2.0；UniverSR 模型及其上游代码采用 MIT。

### N4 — 可测试
- 单元测试覆盖：M/S 编解码、检测各指标、A 路线频谱折叠、评估指标。
- 集成测试：端到端处理一段合成缺损音频，验证修复后指标改善。
- 回归测试：`sidespread eval` 子命令 + 对照数据集。

### N5 — 可配置
- 公开参数：`f_c`（默认 8 kHz）、`R_hf` 阈值（默认 0.3）、输出与报告路径。
- 用户不选择修复模式；所有缺损段使用当前验证效果最好的默认算法。
- 神经模式、相关阈值和 ODE 步数只保留给隐藏的开发评测命令。

## 5. CLI 接口

```
sidespread process <input.wav> [-o <output.wav>]
                  [--fc 8000] [--rhf-threshold 0.3] [--report report.json]

sidespread detect <input.wav> [--fc 8000] [--report report.json]
                  # 仅检测，不处理，输出是否需要修复 + 推荐路线

sidespread eval <clean.wav> [--output <degraded_repaired.wav>]
                  [--fc 8000] [--report report.json]
                  # 合成缺损 → 修复 → 对比原始，输出 ground-truth 评估

sidespread info <input.wav>
                  # 打印 WAV 元数据 + M/S 高频能量概览
```

## 6. 验收标准

1. 输入一段 Suno 风格立体声 WAV，`sidespread detect` 能正确识别是否需要处理。
2. 对需要处理的音频，`sidespread process` 输出修复后的 WAV，缺损段覆盖率接近 100%，`report.json` 显示 LSD/MCD 下降、R_hf 与 ICCC 进入真实音乐参考分布。
3. 对不需要处理的音频，`process` 提示用户并仅输出检测报告。
4. A 路线（DSP）纯 Rust 实现，无外部模型。
5. B 路线（UniverSR ONNX）在 CPU 上能完成推理，输出与 Python 参考实现数值误差 < 1e-3。
6. `sidespread eval` 同时报告参考匹配度、受保护频段 SNR 和感知补足指标；HF-SNR 作为不可恢复信息的诊断值，不要求相对“保持缺失”基线提升。

## 7. 范围外

- 不做音源分离（人声/鼓/贝斯分离）。
- 不做整体音质增强（响度归一、动态范围、去噪）。
- 不做实时流处理（仅离线文件处理）。
- 不训练新模型（使用 UniverSR 预训练权重）。
- 不处理 mono 输入（无 side 可修复）。
