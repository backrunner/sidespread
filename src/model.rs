//! Download and verify the optional UniverSR model.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const MANIFEST_JSON: &str = include_str!("../models/model-manifest.json");

#[derive(Debug, Deserialize)]
struct Manifest {
    models: Models,
}

#[derive(Debug, Deserialize)]
struct Models {
    universr_backbone: ModelInfo,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    url: String,
    sha256: String,
    size_bytes: u64,
}

pub fn download(output: &Path, force: bool) -> Result<()> {
    let model = model_info()?;
    if output.exists() && !force {
        verify_against(output, &model).with_context(|| {
            format!(
                "{} already exists but failed verification; use --force to replace it",
                output.display()
            )
        })?;
        println!("model already verified: {}", output.display());
        return Ok(());
    }

    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating model directory: {}", parent.display()))?;
    }

    let partial = partial_path(output);
    let result = download_to(&model, &partial).and_then(|()| {
        verify_against(&partial, &model)?;
        if output.exists() {
            std::fs::remove_file(output)
                .with_context(|| format!("removing old model: {}", output.display()))?;
        }
        std::fs::rename(&partial, output)
            .with_context(|| format!("installing model: {}", output.display()))?;
        Ok(())
    });
    if result.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
    result?;

    println!("model ready: {}", output.display());
    Ok(())
}

pub fn verify(path: &Path) -> Result<()> {
    let model = model_info()?;
    verify_against(path, &model)?;
    println!("model verified: {}", path.display());
    Ok(())
}

fn model_info() -> Result<ModelInfo> {
    let manifest: Manifest =
        serde_json::from_str(MANIFEST_JSON).context("parsing embedded model manifest")?;
    Ok(manifest.models.universr_backbone)
}

fn download_to(model: &ModelInfo, destination: &Path) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("sidespread/", env!("CARGO_PKG_VERSION")))
        .build()
        .context("creating model download client")?;
    let mut response = client
        .get(&model.url)
        .send()
        .context("downloading UniverSR model")?
        .error_for_status()
        .context("model server returned an error")?;
    let mut file = File::create(destination)
        .with_context(|| format!("creating partial model: {}", destination.display()))?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut downloaded = 0u64;
    let mut last_percentage = None;

    loop {
        let count = response
            .read(&mut buffer)
            .context("reading model response")?;
        if count == 0 {
            break;
        }
        file.write_all(&buffer[..count])
            .context("writing downloaded model")?;
        downloaded += count as u64;
        let percentage = (downloaded.saturating_mul(100) / model.size_bytes).min(100);
        if last_percentage != Some(percentage) {
            eprint!("\rdownloading model: {percentage:3}%");
            std::io::stderr().flush().ok();
            last_percentage = Some(percentage);
        }
    }
    eprintln!();
    file.sync_all().context("flushing downloaded model")?;
    Ok(())
}

fn verify_against(path: &Path, model: &ModelInfo) -> Result<()> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("reading model metadata: {}", path.display()))?;
    if metadata.len() != model.size_bytes {
        bail!(
            "model size mismatch: got {} bytes, expected {}",
            metadata.len(),
            model.size_bytes
        );
    }

    let mut file = File::open(path)
        .with_context(|| format!("opening model for verification: {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).context("hashing model")?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let actual = format!("{:x}", digest.finalize());
    if actual != model.sha256 {
        bail!(
            "model SHA-256 mismatch: got {actual}, expected {}",
            model.sha256
        );
    }
    Ok(())
}

fn partial_path(output: &Path) -> PathBuf {
    let mut path = output.as_os_str().to_os_string();
    path.push(".part");
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_manifest_is_valid() {
        let model = model_info().unwrap();
        assert!(model.url.starts_with("https://assets.sidespread.pwp.sh/"));
        assert_eq!(model.sha256.len(), 64);
        assert!(model.size_bytes > 200_000_000);
    }

    #[test]
    fn partial_path_keeps_the_original_name() {
        assert_eq!(
            partial_path(Path::new("models/universr_backbone.onnx")),
            Path::new("models/universr_backbone.onnx.part")
        );
    }
}
