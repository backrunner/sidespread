//! Sidespread — repair high-frequency loss in the side channel of AI-generated stereo music.

use sidespread::cli;

fn main() -> anyhow::Result<()> {
    cli::run()
}
