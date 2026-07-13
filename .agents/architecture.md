# Sidespread — 架构与模块划分

## 1. 顶层架构

```
┌──────────────────────────────────────────────────────────────┐
│                       CLI (clap)                              │
│  process | detect | eval | info                              │
└──────────┬───────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────┐
│                    Pipeline Orchestrator                      │
│  分段调度：检测 → 路线决策 → 处理 → 拼接 → 重建 → 评估         │
└──────┬───────────┬───────────────┬────────────┬──────────────┘
       │           │               │            │
       ▼           ▼               ▼            ▼
┌──────────┐ ┌──────────┐ ┌─────────────┐ ┌──────────┐
│  IO      │ │ Analysis │ │ Repair      │ │ Eval     │
│  (wav)   │ │ (detect) │ │ (A+B route) │ │ (report) │
└──────────┘ └──────────┘ └─────────────┘ └──────────┘
                              │     │
                              ▼     ▼
                         ┌──────┐ ┌──────────┐
                         │ DSP  │ │ UniverSR │
                         │ (A)  │ │ ONNX (B) │
                         └──────┘ └──────────┘
```

## 2. 目录结构

```
sidespread/
├── Cargo.toml
├── .agents/                      # 设计文档（本目录）
│   ├── README.md
│   ├── requirements.md
│   ├── architecture.md
│   ├── implementation_plan.md
│   └── research_notes.md
├── src/
│   ├── main.rs                   # CLI 入口（clap）
│   ├── cli.rs                    # 命令定义与参数解析
│   ├── pipeline.rs               # Orchestrator：分段调度
│   ├── io/
│   │   ├── mod.rs
│   │   ├── wav.rs                # WAV 读写（hound 封装）
│   │   └── mside.rs              # L/R ↔ M/S 编解码
│   ├── analysis/
│   │   ├── mod.rs
│   │   ├── stft.rs               # STFT / iSTFT（realfft）
│   │   ├── spectrum.rs           # 功率谱、对数谱、mel 谱
│   │   ├── detector.rs           # 缺损检测（R_hf / 谱距离 / 互相关）
│   │   └── segment.rs            # 分段（50-100ms，重叠 50%）
│   ├── repair/
│   │   ├── mod.rs
│   │   ├── dsp.rs                # A 路线：mid→side 高频折叠
│   │   ├── neural.rs             # B 路线：UniverSR ONNX 调用
│   │   ├── universr/
│   │   │   ├── mod.rs
│   │   │   ├── onnx_session.rs   # ort session 加载与管理
│   │   │   ├── frontend.rs       # STFT + mel/complex 前端（对齐 UniverSR）
│   │   │   ├── ode.rs            # 4 步 midpoint ODE 循环（Rust 控制）
│   │   │   ├── istft.rs          # iSTFT 后端（vocoder-free）
│   │   │   └── merge.rs          # 定向频段拼接（原 side 中频 + 网络 high band）
│   │   └── common.rs             # 交叉淡入、相位抖动、限幅
│   ├── eval/
│   │   ├── mod.rs
│   │   ├── metrics.rs            # LSD / MCD / ICCC / R_hf / SNR
│   │   ├── synthetic.rs          # 人为缺损生成（低通 side）
│   │   └── report.rs             # report.json 生成 + 终端摘要
│   └── config.rs                 # 默认参数与 CLI 覆盖
├── models/
│   ├── universr_backbone.onnx    # UniverSR 合并 guided/unconditioned 图（约 232MB）
│   └── universr_config.json
├── tests/
│   ├── io_mside.rs
│   ├── analysis_detector.rs
│   ├── repair_dsp.rs
│   ├── repair_neural.rs
│   └── eval_metrics.rs
└── examples/
    └── (测试用 WAV 样本)
```

## 3. 模块职责

### 3.1 `io` — 音频 I/O 与 M/S 编解码
- `wav.rs`：用 `hound` 读/写 WAV，支持 16/24/32-bit PCM + float，44.1k/48k 立体声。返回 `AudioBuffer { samples: [[f32; N]; 2], sample_rate }`。
- `mside.rs`：`lr_to_ms(L, R) -> (M, S)`、`ms_to_lr(M, S) -> (L, R)`。重建时 soft-clip 限幅。

### 3.2 `analysis` — 检测与判定
- `segment.rs`：把音频按 `segment_ms`（默认 80ms）分段，`overlap`（默认 50%），返回段迭代器。
- `stft.rs`：基于 `realfft` 的 STFT/iSTFT，Hann 窗，`n_fft` / `hop` 可配。
- `spectrum.rs`：功率谱、对数功率谱、归一化谱、mel 滤波器组。
- `detector.rs`：对每段计算 `R_hf`、`LSD_hf`、`cosine_hf`、`corr_hf`，输出 `SegmentReport`：
  ```rust
  struct SegmentReport {
      range: (usize, usize),      // 样本区间
      needs_processing: bool,
      route: Route,               // Skip | Dsp | Neural | Hybrid
      metrics: SegmentMetrics,    // R_hf, LSD, corr 等
  }
  ```

### 3.3 `repair` — 补偿执行
- `dsp.rs`（A 路线）：
  - 输入：M, S, 段区间, 检测指标。
  - 流程：STFT(M), STFT(S) → 对 `[f_c, Nyquist]` bin，`S_mag_new = M_mag * gain_curve`（gain 由 S 中频能量估计），`S_phase_new = M_phase + jitter` → iSTFT → 与原 S 在过渡带交叉淡入。
  - 输出：修复后的 S 段。
- `neural.rs` + `universr/`（B 路线）：
  - `onnx_session.rs`：加载单一 `universr_backbone.onnx`；图返回 guided/unconditioned 两个输出，权重只存储一份，并限制线程/内存缓存控制峰值。
  - `frontend.rs`：把 side 段重采样到 48k（`rubato`），做 UniverSR 期望的 complex STFT 前端（精确对齐 Python 参考的窗/hop/归一化）。
  - `ode.rs`：4 步 midpoint ODE 循环，每步调一次 ONNX forward，CFG combine，更新 latent。
  - `istft.rs`：iSTFT 直出波形（vocoder-free）。
  - `merge.rs`：`S_repaired = S_orig_midband ⊕ S_universr_highband`，过渡带交叉淡入。
- `common.rs`：`crossfade(a, b, fade_len)`、`phase_jitter(shape, max_deg)`、`soft_clip(x)`。

### 3.4 `eval` — 评估与报告
- `metrics.rs`：纯函数，输入两段音频 → 输出 `Metrics { lsd, mcd, iccc, r_hf, snr }`。
- `synthetic.rs`：`degrade_side(M, S, fc) -> S_degraded`（对 S 做 Chebyshev 低通），用于 eval 子命令造对照集。
- `report.rs`：聚合段报告 + 整体指标 → `report.json` + 终端表格。

### 3.5 `pipeline` — 编排
```rust
fn process(input: AudioBuffer, cfg: &Config) -> (AudioBuffer, Report) {
    let (M, S) = lr_to_ms(input);
    let segments = segment(&M, cfg.segment_ms, cfg.overlap);
    let mut S_repaired = S.clone();
    let mut seg_reports = vec![];

    for seg in segments {
        let report = detector::analyze(&M[seg], &S[seg], cfg);
        match report.route {
            Route::Skip => {}
            Route::Dsp => S_repaired[seg] = repair::dsp::process(&M[seg], &S[seg], &report, cfg),
            Route::Neural => S_repaired[seg] = repair::neural::process(&S[seg], cfg),
            Route::Hybrid => {
                let s1 = repair::dsp::process(&M[seg], &S[seg], &report, cfg);
                S_repaired[seg] = repair::neural::refine(&s1, cfg);
            }
        }
        seg_reports.push(report);
    }

    // 段间交叉淡入拼接（避免接缝）
    let S_final = stitch_segments(S_repaired, segments, cfg);
    let out = ms_to_lr(&M, &S_final);
    let overall = eval::metrics::compute(&input, &out, cfg);
    (out, Report { segments: seg_reports, overall })
}
```

### 3.6 `config` — 配置
```rust
struct Config {
    fc: usize,               // 8000
    rhf_threshold: f32,      // 0.3
    corr_high: f32,          // 0.6
    corr_low: f32,           // 0.4
    mode: Mode,              // Auto
    ode_steps: usize,        // 4
    segment_ms: usize,       // 80
    overlap: f32,            // 0.5
    n_fft: usize,            // 4096
    hop: usize,              // 1024
    model_path: PathBuf,     // models/universr_backbone.onnx
    report_path: PathBuf,
}
```

## 4. 数据流

```
WAV file
  │
  ▼ hound::read
AudioBuffer { L, R, sr }
  │
  ▼ lr_to_ms
M, S  (f32 vectors)
  │
  ▼ segment + detector
Vec<SegmentReport>   (每段：needs_processing? route?)
  │
  ▼ 按路由分发
  ├── Dsp route ──→ repair::dsp (STFT 折叠) ──→ S_seg_repaired
  ├── Neural route ──→ repair::neural (UniverSR ONNX) ──→ S_seg_repaired
  └── Skip ──→ S_seg 原样
  │
  ▼ stitch (段间交叉淡入)
S_final
  │
  ▼ ms_to_lr + soft_clip
AudioBuffer { L', R', sr }
  │
  ▼ eval::metrics (vs 原始)
Report { segments, overall_metrics }
  │
  ▼ hound::write + report.json
output.wav + report.json
```

## 5. 关键依赖

| crate | 用途 |
|---|---|
| `hound` | WAV 读写 |
| `realfft` | 实数 FFT（STFT/iSTFT） |
| `rustfft` | 复数 FFT（备用） |
| `ndarray` | 矩阵/张量运算 |
| `ort` | ONNX Runtime（UniverSR 推理），通过 `ep` 模块统一调度 CPU/CUDA/ROCm/CoreML/DirectML |
| `rubato` | 重采样（48k 对齐） |
| `clap` | CLI |
| `serde` + `serde_json` | report.json |
| `apodize` | 窗函数（Hann 等） |

## 6. 错误处理策略
- WAV 解析错误、单声道输入 → `anyhow::Error`，CLI 友好提示退出。
- ONNX session 加载失败 → 提示模型路径错误或需下载（给出下载脚本）。
- 数值 NaN/Inf → 检测并报错，避免输出静音/爆音。
- 评估指标计算失败（如某段全静音）→ 该段标记 N/A，不中断流程。

## 7. 线程与性能
- A 路线：段间可并行（`rayon`），无状态。
- B 路线：ONNX session 共享，`ort` 内部管理线程池；段间串行（避免内存峰值）。
- 大文件：流式分段处理，不一次性加载全部 PCM。
