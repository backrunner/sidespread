//! Small, dependency-free terminal presentation helpers.
//!
//! Sidespread is also used from scripts, so color is enabled only for an
//! interactive terminal (or when `FORCE_COLOR` is set) and is disabled by
//! `NO_COLOR`.

use std::fmt::Display;
use std::io::{self, IsTerminal, Write};

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

pub fn progress(label: &str, current: u64, total: u64) {
    let ratio = if total == 0 {
        0.0
    } else {
        (current as f64 / total as f64).clamp(0.0, 1.0)
    };
    let width = 28usize;
    let filled = (ratio * width as f64).round() as usize;
    let bar = format!(
        "{}{}",
        "=".repeat(filled),
        ".".repeat(width.saturating_sub(filled))
    );
    let line = format!(
        "\r{} {:>3}% [{bar}] {}/{} MB",
        label,
        (ratio * 100.0).round() as u8,
        current / 1_000_000,
        total / 1_000_000
    );
    if color_enabled() {
        eprint!("\x1b[36m{line}\x1b[0m");
    } else {
        eprint!("{line}");
    }
    io::stderr().flush().ok();
}

pub fn finish_progress() {
    eprintln!();
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

fn is_confirmation(answer: &str) -> bool {
    let answer = answer.trim().to_ascii_lowercase();
    answer.is_empty() || matches!(answer.as_str(), "y" | "yes")
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
}
