//! Small terminal presentation helpers.
//!
//! Sidespread is also used from scripts, so color is enabled only for an
//! interactive terminal (or when `FORCE_COLOR` is set) and is disabled by
//! `NO_COLOR`.

use std::fmt::Display;
use std::io::{self, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use terminal_size::{terminal_size_of, Width};

#[derive(Clone, Copy)]
pub enum Tone {
    Cyan,
    Green,
    Yellow,
    Red,
    Blue,
    Muted,
    White,
}

impl Tone {
    fn code(self) -> &'static str {
        match self {
            Self::Cyan => "36",
            Self::Green => "32",
            Self::Yellow => "33",
            Self::Red => "31",
            Self::Blue => "34",
            Self::Muted => "90",
            Self::White => "37",
        }
    }
}

pub fn color_enabled() -> bool {
    if std::env::var_os("NO_COLOR").is_some() {
        return false;
    }
    if let Ok(value) = std::env::var("FORCE_COLOR") {
        return value != "0" && !value.eq_ignore_ascii_case("false");
    }
    io::stdout().is_terminal() && io::stderr().is_terminal()
}

pub fn interactive() -> bool {
    io::stdin().is_terminal() && io::stderr().is_terminal()
}

pub fn progress_ui_enabled() -> bool {
    io::stderr().is_terminal()
        && std::env::var_os("TERM").is_none_or(|term| term != "dumb")
        && std::env::var_os("CI").is_none()
}

pub fn paint(value: impl Display, tone: Tone) -> String {
    let value = value.to_string();
    if color_enabled() {
        format!("\x1b[{}m{value}\x1b[0m", tone.code())
    } else {
        value
    }
}

pub fn rule() -> String {
    paint(
        "+------------------------------------------------------------------+",
        Tone::Muted,
    )
}

pub fn header(title: &str, subtitle: &str) {
    println!("{}", rule());
    println!(
        "| {} {} |",
        paint(format!("{title:<14}"), Tone::Cyan),
        paint(format!("{subtitle:<49}"), Tone::White)
    );
    println!("{}", rule());
}

pub fn section(title: &str) {
    println!();
    println!("{}", paint(format!("[ {title} ]"), Tone::Blue));
}

pub fn status(label: &str, detail: impl Display, tone: Tone) {
    eprintln!("{} {}", paint(format!("[{label}]"), tone), detail);
}

pub fn error_report(error: &anyhow::Error) {
    eprintln!();
    eprintln!("{}", rule());
    eprintln!(
        "| {} |",
        paint(format!("{:<64}", "SIDESPREAD ERROR"), Tone::Red)
    );
    eprintln!("{}", rule());
    eprintln!("  {}", paint(error, Tone::White));
    let causes = error.chain().skip(1).collect::<Vec<_>>();
    if !causes.is_empty() {
        eprintln!("  {}", paint("Caused by", Tone::Muted));
        for (index, cause) in causes.iter().enumerate() {
            eprintln!(
                "  {} {cause}",
                paint(format!("{:>2}.", index + 1), Tone::Red)
            );
        }
    }
    eprintln!("{}", rule());
}

pub fn route_label(route: impl Display) -> String {
    let route = route.to_string();
    let tone = match route.trim() {
        "skip" => Tone::Muted,
        "dsp" => Tone::Cyan,
        "neural" => Tone::Yellow,
        "hybrid" => Tone::Green,
        _ => Tone::White,
    };
    paint(route, tone)
}

/// A compact, thread-safe progress row for interactive processing. It is silent for pipes and CI.
pub struct TaskProgress {
    label: String,
    unit: &'static str,
    total: u64,
    current: AtomicU64,
    last_percentage: AtomicU64,
    last_render_ms: AtomicU64,
    started: Instant,
    finished: AtomicBool,
    draw_lock: Mutex<()>,
    line_width: usize,
    visible: bool,
    enabled: bool,
}

impl TaskProgress {
    pub fn new(label: impl Into<String>, total: usize, unit: &'static str) -> Self {
        Self::new_if(label, total, unit, true)
    }

    pub(crate) fn new_if(
        label: impl Into<String>,
        total: usize,
        unit: &'static str,
        visible: bool,
    ) -> Self {
        let progress = Self {
            label: label.into(),
            unit,
            total: total as u64,
            current: AtomicU64::new(0),
            last_percentage: AtomicU64::new(u64::MAX),
            last_render_ms: AtomicU64::new(0),
            started: Instant::now(),
            finished: AtomicBool::new(false),
            draw_lock: Mutex::new(()),
            line_width: progress_line_width(),
            visible,
            enabled: visible && progress_ui_enabled(),
        };
        progress.render(true);
        progress
    }

    pub fn advance(&self, amount: usize) {
        if self.finished.load(Ordering::Relaxed) {
            return;
        }
        self.current.fetch_add(amount as u64, Ordering::Relaxed);
        self.render(false);
    }

    pub fn set(&self, current: usize) {
        if self.finished.load(Ordering::Relaxed) {
            return;
        }
        self.current.store(current as u64, Ordering::Relaxed);
        self.render(false);
    }

    pub fn finish(&self) {
        if self.finished.swap(true, Ordering::Relaxed) {
            return;
        }
        self.current.store(self.total, Ordering::Relaxed);
        self.render(true);
        if self.enabled {
            eprintln!();
        } else if self.visible {
            status(&self.label, "complete", Tone::Green);
        }
    }

    fn render(&self, force: bool) {
        if !self.enabled {
            return;
        }
        let elapsed_ms = self.started.elapsed().as_millis().min(u64::MAX as u128) as u64;
        let last_render_ms = self.last_render_ms.load(Ordering::Relaxed);
        if !force && elapsed_ms.saturating_sub(last_render_ms) < 500 {
            return;
        }
        let _guard = self
            .draw_lock
            .lock()
            .expect("progress lock is not poisoned");
        let elapsed_ms = self.started.elapsed().as_millis().min(u64::MAX as u128) as u64;
        let current = self.current.load(Ordering::Relaxed).min(self.total);
        let percentage = current
            .saturating_mul(100)
            .checked_div(self.total)
            .unwrap_or(100);
        let last_render_ms = self.last_render_ms.load(Ordering::Relaxed);
        if !force && percentage < 100 && elapsed_ms.saturating_sub(last_render_ms) < 500 {
            return;
        }
        if !force && self.last_percentage.load(Ordering::Relaxed) == percentage {
            return;
        }
        self.last_percentage.store(percentage, Ordering::Relaxed);
        self.last_render_ms.store(elapsed_ms, Ordering::Relaxed);

        let elapsed = self.started.elapsed();
        let line = progress_line(
            &self.label,
            self.unit,
            current,
            self.total,
            elapsed,
            self.line_width,
        );
        if color_enabled() {
            eprint!("\r\x1b[2K\x1b[36m{line}\x1b[0m");
        } else {
            eprint!("\r\x1b[2K{line}");
        }
        io::stderr().flush().ok();
    }
}

fn progress_line_width() -> usize {
    let columns = terminal_size_of(io::stderr())
        .map(|(Width(columns), _)| usize::from(columns))
        .or_else(|| {
            std::env::var("COLUMNS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
        })
        .unwrap_or(80)
        .clamp(20, 240);
    // Avoid the terminal's deferred-wrap state at the final visible column.
    columns.saturating_sub(1)
}

fn progress_line(
    label: &str,
    unit: &str,
    current: u64,
    total: u64,
    elapsed: Duration,
    max_width: usize,
) -> String {
    let percentage = current
        .saturating_mul(100)
        .checked_div(total)
        .unwrap_or(100);
    let elapsed_text = format_duration(elapsed);
    let remaining = (current > 0 && current < total).then(|| {
        let seconds = elapsed.as_secs_f64() * (total - current) as f64 / current as f64;
        format_duration(Duration::from_secs_f64(seconds))
    });
    let timing = remaining
        .map(|remaining| format!("{elapsed_text}/{remaining} left"))
        .unwrap_or_else(|| elapsed_text.clone());
    let prefix = format!("[{label}] ");
    let detailed_suffix = format!(" {percentage:>3}% {current}/{total} {unit} | {timing}");
    let detailed_bar_width = max_width
        .saturating_sub(prefix.len() + detailed_suffix.len() + 2)
        .min(26);
    if detailed_bar_width >= 8 {
        return format!(
            "{prefix}[{}{}]{detailed_suffix}",
            "=".repeat(percentage as usize * detailed_bar_width / 100),
            ".".repeat(detailed_bar_width - percentage as usize * detailed_bar_width / 100)
        );
    }

    let medium_suffix = format!(" {percentage:>3}% | {elapsed_text}");
    let medium_bar_width = max_width
        .saturating_sub(prefix.len() + medium_suffix.len() + 2)
        .min(18);
    if medium_bar_width >= 6 {
        return format!(
            "{prefix}[{}{}]{medium_suffix}",
            "=".repeat(percentage as usize * medium_bar_width / 100),
            ".".repeat(medium_bar_width - percentage as usize * medium_bar_width / 100)
        );
    }

    let compact_suffix = format!(" {percentage:>3}% {elapsed_text}");
    let label_width = max_width.saturating_sub(compact_suffix.len() + 2);
    let label = label.chars().take(label_width).collect::<String>();
    format!("[{label}]{compact_suffix}")
}

impl Drop for TaskProgress {
    fn drop(&mut self) {
        if self.enabled && !self.finished.load(Ordering::Relaxed) {
            eprintln!();
        }
    }
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs();
    if seconds >= 3_600 {
        format!("{}h{:02}m", seconds / 3_600, (seconds % 3_600) / 60)
    } else if seconds >= 60 {
        format!("{}m{:02}s", seconds / 60, seconds % 60)
    } else {
        format!("{seconds}s")
    }
}

/// Ask a question only when both streams are interactive. This keeps library
/// calls and CI jobs non-blocking while providing a friendly first-run flow.
pub fn confirm(question: &str) -> io::Result<bool> {
    if !interactive() {
        return Ok(false);
    }
    eprint!("{} ", paint(question, Tone::Yellow));
    io::stderr().flush()?;
    let mut answer = String::new();
    if io::stdin().read_line(&mut answer)? == 0 {
        return Ok(false);
    }
    Ok(is_confirmation(&answer))
}

/// Ask an interactive question whose empty/default answer is no.
pub fn confirm_default_no(question: &str) -> io::Result<bool> {
    if !interactive() {
        return Ok(false);
    }
    eprint!("{} ", paint(question, Tone::Yellow));
    io::stderr().flush()?;
    let mut answer = String::new();
    if io::stdin().read_line(&mut answer)? == 0 {
        return Ok(false);
    }
    Ok(is_explicit_confirmation(&answer))
}

fn is_confirmation(answer: &str) -> bool {
    let answer = answer.trim().to_ascii_lowercase();
    answer.is_empty() || matches!(answer.as_str(), "y" | "yes")
}

fn is_explicit_confirmation(answer: &str) -> bool {
    matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_labels_keep_the_original_text() {
        assert!(route_label("dsp").contains("dsp"));
    }

    #[test]
    fn confirmation_defaults_to_yes() {
        assert!(is_confirmation(""));
        assert!(is_confirmation("Y"));
        assert!(is_confirmation("yes"));
        assert!(!is_confirmation("n"));
    }

    #[test]
    fn explicit_confirmation_defaults_to_no() {
        assert!(!is_explicit_confirmation(""));
        assert!(is_explicit_confirmation("Y"));
        assert!(is_explicit_confirmation("yes"));
        assert!(!is_explicit_confirmation("n"));
    }

    #[test]
    fn progress_rows_never_exceed_the_available_width() {
        for width in [20, 39, 55, 79, 119] {
            let line = progress_line(
                "HARMONIC DEBLEED",
                "frames",
                4_712,
                12_345,
                Duration::from_secs(73),
                width,
            );
            assert!(
                line.len() <= width,
                "width={width}, rendered={}, line={line:?}",
                line.len()
            );
            assert!(line.contains('%'));
        }
    }
}
