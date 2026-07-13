# Sidespread — 实现方案与计划

## 1. 实现阶段总览

| 阶段 | 内容 | 工时 | 依赖 |
|---|---|---|---|
| **P0** | 项目骨架 + CLI + WAV/M-S I/O | 1 天 | 无 |
| **P1** | 检测模块（STFT + 缺损检测） | 1.5 天 | P0 |
| **P2** | A 路线 DSP 补偿（纯 Rust） | 2 天 | P1 |
| **P3** | 评估模块（LSD/MCD/ICCC/报告） | 1 天 | P1 |
| **P4** | UniverSR ONNX 导出（Python 端） | 2 天 | 无（可与 P1-P3 并行） |
| **P5** | B 路线 Rust 集成（ort + 前端 + ODE + iSTFT） | 4 天 | P1, P4 |
| **P6** | 端到端集成 + 段拼接 + 重建 | 1 天 | P2, P3, P5 |
| **P7** | 对照测试集 + eval 子命令 + 回归 | 1.5 天 | P6 |
| **P8** | 性能调优 + 文档 + 打包 | 1 天 | P7 |
| **合计** | | **~15 天** | |

## 2. P0 — 项目骨架（1 天）

### 交付物
- `Cargo.toml`，依赖：`hound, realfft, ndarray, clap, serde, serde_json, anyhow, rubato`（`ort` 在 P5 加）。
- `src/main.rs` + `cli.rs`：四个子命令骨架（process/detect/eval/info），参数解析，未实现部分返回 `unimplemented`。
- `src/io/wav.rs`：`read_wav(path) -> AudioBuffer`、`write_wav(path, buf)`。
- `src/io/mside.rs`：`lr_to_ms` / `ms_to_lr` + 单声道检测报错。
- `src/config.rs`：`Config` 结构 + 默认值 + CLI 覆盖。
- `sidespread info <wav>` 能跑通，打印采样率/位深/时长/M-S 高频能量概览。

### 验证
- 读一个真实立体声 WAV，打印 L/R 和 M/S 的 RMS。
- 写回一个 WAV，用 `ffmpeg`/`sox` 验证可播放。

## 3. P1 — 检测模块（1.5 天）

### 交付物
- `src/analysis/stft.rs`：`StftConfig { n_fft, hop, window }`，`stft(signal, cfg) -> Vec<SpectrumFrame>`，`istft(frames, cfg) -> signal`（overlap-add）。Hann 窗，`n_fft=4096`，`hop=1024`。
- `src/analysis/spectrum.rs`：功率谱、对数谱、归一化、mel 滤波器组（预计算 mel basis 常量）。
- `src/analysis/segment.rs`：分段迭代器。
- `src/analysis/detector.rs`：
  ```rust
  fn analyze(m_seg: &[f32], s_seg: &[f32], cfg: &Config) -> SegmentReport {
      let m_spec = stft(m_seg, ...);
      let s_spec = stft(s_seg, ...);
      let r_hf = high_freq_energy_ratio(&m_spec, &s_spec, cfg.fc);
      let lsd_hf = log_spectral_distance_hf(&m_spec, &s_spec, cfg.fc);
      let corr_hf = cross_correlation_hf(&m_spec, &s_spec, cfg.fc);
      let (needs, route) = decide(r_hf, corr_hf, cfg);
      SegmentReport { ... }
  }
  ```

### 验证
- 拿一个干净立体声 + 人为低通 side 的版本，检测器应输出：干净→`Skip`，缺损→`Dsp` 或 `Neural`。
- 单元测试：`r_hf` 对静音 side = 0，对等能量 = 1。

## 4. P2 — A 路线 DSP 补偿（2 天）

### 交付物
- `src/repair/dsp.rs`：
  ```rust
  pub fn repair(m_seg: &[f32], s_seg: &[f32], report: &SegmentReport, cfg: &Config)
      -> Vec<f32>  // 修复后的 S 段
  ```
  流程：
  1. `M_spec = stft(m_seg)`，`S_spec = stft(s_seg)`。
  2. 对 bin `b >= bin(fc)`：
     - `gain[b] = estimate_gain_from_midband(S_spec, M_spec, b)`（从 S 中频能量曲线外推）。
     - `S_mag_new[b] = M_spec.mag[b] * gain[b]`。
     - `S_phase_new[b] = M_spec.phase[b] + phase_jitter(b)`（±5°~30°，随 bin 平滑变化）。
  3. 过渡带 `[fc - transition, fc + transition]`（默认 transition = 500 Hz）：`S_new = blend(S_orig, S_repaired, smoothstep)`。
  4. `s_repaired = istft(S_spec_new)`。
  5. 与原 `s_seg` 在段边界交叉淡入。
- `src/repair/common.rs`：`crossfade`、`phase_jitter`、`soft_clip`、`smoothstep`。

### 验证
- 对"干净→低通 side"的合成样本，A 路线修复后 `R_hf` 应从 ~0.1 升到 >0.6，LSD_hf 下降。
- 听感：高频立体声宽度恢复，无明显伪影。
- 单元测试：mid=side 时（完全相关），A 路线应基本无副作用。

## 5. P3 — 评估模块（1 天）

### 交付物
- `src/eval/metrics.rs`：
  - `lsd(a_spec, b_spec) -> f32`（对数谱距离）。
  - `mcd(a_mel, b_mel) -> f32`（mel 倒谱距离）。
  - `iccc(a_high, b_high) -> f32`（高频互相关）。
  - `r_hf(m_spec, s_spec, fc) -> f32`。
  - `snr(original, repaired) -> f32`（用于 eval 子命令有 ground truth 时）。
- `src/eval/report.rs`：`Report { segments, overall }` → `serde_json` 输出 + 终端表格（`prettytable` 或手写对齐）。
- `src/eval/synthetic.rs`：`degrade_side(M, S, fc) -> S_degraded`（Chebyshev type-II 低通，用 `biquad` crate 或自实现 sosfiltfilt）。

### 验证
- 对已知缺损样本，metrics 输出合理数值。
- report.json 可被 `jq` 解析。

## 6. P4 — UniverSR ONNX 导出（2 天，Python 端）

### 目标
把 UniverSR 的推理图导出为单一 ONNX 文件，可在 Rust `ort` 中加载。

### 步骤
1. **clone + 跑通 Python 参考**：
   - `git clone https://github.com/woongzip1/UniverSR`
   - 下载权重 `huggingface.co/woongzip1/universr-audio`（`pytorch_model.bin` 229MB）。
   - 在一段音乐 side 通道上跑 Python 推理，保存输出作 Rust 对照基准。
2. **识别前端/后端**：读 `infer.py` / `model.py`，记录：
   - 输入 STFT 参数（n_fft, hop, win, window type, 是否 center, 归一化）。
   - 输入 mel/complex 表示（real/imag 双通道？magnitude+phase？）。
   - 输出 iSTFT 参数。
   - CFG（classifier-free guidance）的 combine 方式。
   - ODE solver（midpoint, 4 步）的更新公式。
3. **导出 ONNX**：
   - 单图导出：把 ConvNeXt-V2 backbone 的单步 forward 导为一个 ONNX。
     ```python
     torch.onnx.export(
         model.net,                         # 单步网络
         (lr_complex_stft,),                # 示例输入
         "universr_backbone.onnx",
         dynamo=True,
         input_names=["lr_stft"],
         output_names=["pred_stft"],
     )
     ```
   - 固定 chunk shape（UniverSR 的 chunk 长度需从代码确认，预估 ~2-5s）。
   - 验证 ONNX 输出与 PyTorch 输出误差 < 1e-5。
4. **记录所有"图外"步骤**：STFT 前端、iSTFT 后端、CFG combine、ODE 循环、重采样——这些在 Rust 里复刻，不进 ONNX。

### 验证
- `onnxruntime` Python 加载 ONNX，输出与 PyTorch 一致。
- 记录一份 `universr_pipeline_spec.md`：前端参数 + ODE 公式 + 后端参数，供 Rust 实现对照。

## 7. P5 — B 路线 Rust 集成（4 天）

### 交付物
- `src/repair/neural.rs`：
  ```rust
  pub fn repair(s_seg: &[f32], cfg: &Config) -> Vec<f32>
  ```
  顶层流程：
  1. 重采样到 48k（`rubato`）。
  2. `frontend::stft_to_universr_input(s_48k)` → complex STFT 表示。
  3. `ode::solve(initial, steps=4, |x| onnx_forward(x))` → 修复后的 complex STFT。
  4. `istft::to_waveform( repaired_stft)` → 全频段 side。
  5. `merge::band_merge(s_orig, s_fullband, fc)` → 只取高频段，中频保留原始。
  6. 重采样回原采样率（若输入非 48k）。
- `src/repair/universr/onnx_session.rs`：单例 `ort::Session`，`once_cell` 或 `OnceLock`。通过 `ort::execution_providers()` 按可用性依次尝试 CUDA / ROCm / CoreML / CPU（`ep` 模块），自动选择当前硬件最优 EP，单一代码路径覆盖全平台。
- `src/repair/universr/frontend.rs`：精确复刻 Python STFT（窗、hop、center、归一化）。**这是最容易翻车的点**——逐 bin 与 Python 输出对比验证。
- `src/repair/universr/ode.rs`：
  ```rust
  fn midpoint_solve<F>(x0: Tensor, steps: usize, f: F) -> Tensor
  where F: Fn(&Tensor) -> Tensor  // f = ONNX forward + CFG
  {
      let mut x = x0;
      for _ in 0..steps {
          let k1 = f(&x);
          let x_mid = x + 0.5 * dt * k1;   // dt 由 UniverSR 配置确定
          let k2 = f(&x_mid);
          x = x + dt * k2;
      }
      x
  }
  ```
  （实际公式以 UniverSR 代码为准，可能涉及 v-prediction / x0-prediction 转换）
- `src/repair/universr/istft.rs`：iSTFT，overlap-add。
- `src/repair/universr/merge.rs`：`band_merge(orig, repaired, fc, transition)`，频域 mask 平滑过渡（参考 LavaSR 的 Linkwitz-Riley 思路，但这里用更简单的 smoothstep mask）。

### 验证（关键里程碑）
- **前端数值对齐**：Rust STFT 输出 vs Python STFT 输出，max abs diff < 1e-4。不过这关，后续全废。
- **单步 ONNX 输出对齐**：Rust `ort` forward vs Python `onnxruntime` forward，max abs diff < 1e-4。
- **端到端对齐**：Rust 完整 B 路线输出 vs Python UniverSR 输出，SNR > 40 dB（允许 1e-3 级累积误差）。
- **音乐质量**：对合成缺损样本，B 路线修复后 LSD_hf 下降，听感自然。

## 8. P6 — 端到端集成（1 天）

### 交付物
- `src/pipeline.rs`：完整 orchestrator（架构文档 §3.5 的伪码落地）。
- 段间交叉淡入拼接：相邻段在重叠区用 Hann 窗加权求和。
- `ms_to_lr` 重建 + soft-clip。
- `process` / `detect` / `eval` 子命令全部接通。
- 整体不需要处理的音频 → 提示用户 + 仅输出检测报告。

### 验证
- 端到端跑一首完整 Suno 风格曲目，输出修复 WAV + report.json。
- 段间无接缝爆音。

## 9. P7 — 对照测试集 + 回归（1.5 天）

### 交付物
- `tests/` 下集成测试：
  - 取 3-5 段干净立体声音乐（不同风格：电子/人声/古典）。
  - `synthetic::degrade_side` 生成缺损版本。
  - 跑 `process`，断言 `overall.r_hf` 提升、`overall.lsd` 下降。
  - 跑 `eval`，断言 `snr_vs_ground_truth` > 某阈值。
- `sidespread eval` 子命令完整可用。
- 一份基准报告：A 路线 vs B 路线 vs 混合，在对照集上的指标对比。

### 验证
- `cargo test` 全绿。
- 回归基线指标记录到 `.agents/benchmark_baseline.md`（供后续优化对比）。

## 10. P8 — 调优与收尾（1 天）

- `rayon` 并行化 A 路线段处理。
- ONNX session 预热（首次推理慢）。
- 大文件流式处理（避免全量 PCM 入内存）。
- README（用户文档）+ `--help` 文案打磨。
- 模型下载脚本 `scripts/download_model.sh`。
- `cargo build --release` 产出单二进制。

## 11. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| UniverSR 前端 STFT 复刻不精确 | 中 | 高（latent 垃圾） | P5 第一天就做 Python-vs-Rust 逐 bin 对齐，不过关不往下走 |
| ONNX 导出图断裂 / 动态形状 | 中 | 中 | 固定 chunk shape；dynamo 导出失败时回退 legacy export 或分图 |
| CPU 推理太慢 | 低 | 中 | 先实测；不行降 `ode_steps=2` 或换 `euler` solver |
| A 路线相位伪影 | 中 | 低 | 相位抖动 + 过渡带平滑 + 听感验证 |
| 段间接缝 | 中 | 低 | 50% 重叠 + Hann 交叉淡入 |
| UniverSR 对 side 单通道 OOD | 低 | 中 | UniverSR 训练含音乐 460h，side 本质是音乐差信号，分布接近 |

## 12. 关键决策记录

- **B 路线模型选 UniverSR**（非 FlashSR）：音乐 benchmark 全面更优（2f-model +5~7 分，LSD-HF -25~-40%），MIT 许可，229MB 单图，全卷积 + iSTFT 直出更易导 ONNX。
- **定向频段补齐**而非整段超分：保留 side 原始中频，只用网络补高频，与需求完全对齐且降低副作用。
- **ODE 循环在 Rust 控制**而非烘焙进 ONNX：灵活性高，可调步数/solver，单图导出更稳。
- **A 路线纯 Rust**：mid/S 相关时用，廉价保真兜底，无需模型。
- **推理运行时选 `ort`（ONNX Runtime）而非 tch-rs**：tch-rs 在 ROCm 上 stock 0.24.0 静默回退 CPU（issue #1015 无 fix），MPS 上多个 panic bug（#687/#773/#777），且 libtorch 1.7GB vs ort 50MB。ort 的 `ep` 模块官方覆盖 CPU/CUDA/ROCm/CoreML/DirectML，单代码路径全平台。UniverSR 全卷积 ONNX 友好，不需 tch-rs 的 autograd 优势。
