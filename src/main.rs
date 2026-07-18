//! Sidespread — repair high-frequency loss in the side channel of AI-generated stereo music.

use sidespread::cli;

fn main() {
    if let Err(error) = cli::run() {
        sidespread::terminal::error_report(&error);
        std::process::exit(1);
    }
}
