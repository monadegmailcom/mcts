//! `.safetensors` checkpoints: where they live and which network they hold.
//!
//! The Browser Lab and its HTTP API name checkpoints by plain file name
//! (`uttt_exp4_block.safetensors`); this module maps those names to one folder: `checkpoints/`
//! under the working directory, or `$MCTS_CHECKPOINT_DIR` (e.g. a mounted volume in a container).
//!
//! Every checkpoint records its network architecture ([`AlphaZeroNetConfig`] as JSON) in the
//! safetensors metadata, so loading needs no architecture choice; [`net_config`] reads it, and
//! for files written before the metadata existed finds the game preset whose tensors match.

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

use crate::alphazero::AlphaZeroGame;
use crate::nn::{AlphaZeroNet, AlphaZeroNetConfig};

pub const EXTENSION: &str = ".safetensors";

pub fn dir() -> PathBuf {
    std::env::var_os("MCTS_CHECKPOINT_DIR").map_or_else(|| PathBuf::from("checkpoints"), PathBuf::from)
}

/// Path of the checkpoint named `name`, creating the folder if needed (a failure to create it
/// surfaces when the file is written).
pub fn path(name: &str) -> PathBuf {
    let dir = dir();
    let _ = std::fs::create_dir_all(&dir);
    dir.join(name)
}

/// Turns a user-supplied name into a safe checkpoint file name with the extension: only ASCII
/// letters, digits, `_`, `-` and `.`, no hidden files and no `..`. `None` if nothing usable remains.
pub fn sanitize_name(raw: &str) -> Option<String> {
    let clean: String =
        raw.trim().chars().filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')).collect();
    if clean.is_empty() || clean.starts_with('.') || clean.contains("..") {
        return None;
    }
    Some(if clean.ends_with(EXTENSION) { clean } else { format!("{clean}{EXTENSION}") })
}

/// `(file name, size in bytes)` of every checkpoint in the folder, sorted by name.
pub fn list() -> Vec<(String, u64)> {
    let Ok(entries) = std::fs::read_dir(dir()) else { return Vec::new() };
    let mut files: Vec<(String, u64)> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            name.ends_with(EXTENSION).then(|| (name, e.metadata().map_or(0, |m| m.len())))
        })
        .collect();
    files.sort();
    files
}

/// Metadata key holding the network architecture.
const NET_CONFIG_KEY: &str = "net_config";

/// Splits a safetensors file into its JSON header and the tensor data after it.
fn read_header(path: &Path) -> io::Result<(serde_json::Value, Vec<u8>)> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    let invalid = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {msg}", path.display()));
    let len = u64::from_le_bytes(bytes.get(..8).ok_or_else(|| invalid("too short"))?.try_into().expect("8 bytes")) as usize;
    let header = bytes.get(8..8 + len).ok_or_else(|| invalid("truncated header"))?;
    let json = serde_json::from_slice(header).map_err(|e| invalid(&e.to_string()))?;
    Ok((json, bytes.split_off(8 + len)))
}

/// Records `config` in the metadata of the safetensors file at `path` (rewrites the header; the
/// tensor data is unchanged, since tensor offsets are relative to the end of the header).
pub fn write_net_config(path: &Path, config: &AlphaZeroNetConfig) -> io::Result<()> {
    let (mut header, data) = read_header(path)?;
    let config_json = serde_json::to_string(config).map_err(io::Error::other)?;
    let meta = header
        .as_object_mut()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "safetensors header is not an object"))?
        .entry("__metadata__")
        .or_insert_with(|| serde_json::json!({}));
    meta[NET_CONFIG_KEY] = serde_json::Value::String(config_json);
    let mut header_bytes = serde_json::to_vec(&header).map_err(io::Error::other)?;
    // Keep the tensor data 8-byte aligned, as safetensors writers do (padding with spaces).
    header_bytes.resize(header_bytes.len().div_ceil(8) * 8, b' ');
    let tmp = path.with_extension("safetensors.tmp");
    let mut file = File::create(&tmp)?;
    file.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    file.write_all(&header_bytes)?;
    file.write_all(&data)?;
    drop(file);
    std::fs::rename(tmp, path)
}

/// The architecture recorded in the checkpoint's metadata (`None` for files without it).
pub fn read_net_config(path: &Path) -> Option<AlphaZeroNetConfig> {
    let (header, _) = read_header(path).ok()?;
    serde_json::from_str(header["__metadata__"][NET_CONFIG_KEY].as_str()?).ok()
}

/// The architecture of the checkpoint at `path`: from its metadata, or, for files written before
/// the metadata existed, the first of `R`'s presets whose parameter names and shapes match the file.
pub fn net_config<R: AlphaZeroGame<G, P, N>, const G: usize, const P: usize, const N: usize>(
    path: &Path,
) -> Result<AlphaZeroNetConfig, String> {
    if !path.exists() {
        return Err(format!("checkpoint {} does not exist", path.display()));
    }
    if let Some(config) = read_net_config(path) {
        return Ok(config);
    }
    R::net_presets()
        .into_iter()
        .map(|(_, config)| config)
        .find(|config| {
            let mut varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
            AlphaZeroNet::<G, P, N>::new(vb, config).is_ok() && varmap.load(path).is_ok()
        })
        .ok_or_else(|| format!("{} matches none of the game's network presets", path.display()))
}

/// Converts the block-network checkpoint `src`, whose policy head shares one context among all
/// blocks, into the same network with one context per block (`per_block_context`) and writes it to
/// `dst`. Every block starts with a copy of the shared projection, so the result computes exactly
/// what `src` computes and can then learn block-specific contexts, e.g. to continue a run with the
/// wider head. Returns the new architecture.
pub fn widen_policy_context<R: AlphaZeroGame<G, P, N>, const G: usize, const P: usize, const N: usize>(
    src: &Path,
    dst: &Path,
) -> Result<AlphaZeroNetConfig, String> {
    let mut config = net_config::<R, G, P, N>(src)?;
    let AlphaZeroNetConfig::BlockNet { board_height, board_width, block_height, block_width, per_block_context, .. } =
        &mut config
    else {
        return Err(format!("{} is not a block network", src.display()));
    };
    if *per_block_context {
        return Err(format!("{} already has one policy context per block", src.display()));
    }
    *per_block_context = true;
    let blocks = (*board_height / *block_height) * (*board_width / *block_width);

    let mut tensors = candle_core::safetensors::load(src, &Device::Cpu).map_err(|e| e.to_string())?;
    for name in ["policy_ctx.weight", "policy_ctx.bias"] {
        let shared = tensors.get(name).ok_or_else(|| format!("{} has no {name}", src.display()))?;
        // Output rows are block-major: rows `k * local_dim..(k + 1) * local_dim` are block `k`'s.
        let tiled = Tensor::cat(&vec![shared; blocks], 0).map_err(|e| e.to_string())?;
        tensors.insert(name.to_string(), tiled);
    }
    candle_core::safetensors::save(&tensors, dst).map_err(|e| e.to_string())?;
    write_net_config(dst, &config).map_err(|e| e.to_string())?;
    Ok(config)
}

/// Name of the game preset with this architecture, or `"custom"`.
pub fn preset_name<R: AlphaZeroGame<G, P, N>, const G: usize, const P: usize, const N: usize>(
    config: &AlphaZeroNetConfig,
) -> &'static str {
    R::net_presets().into_iter().find(|(_, c)| c == config).map_or("custom", |(name, _)| name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_sanitized() {
        assert_eq!(sanitize_name(" uttt_best "), Some("uttt_best.safetensors".into()));
        assert_eq!(sanitize_name("ttt_a.safetensors"), Some("ttt_a.safetensors".into()));
        assert_eq!(sanitize_name("../etc/passwd"), None);
        assert_eq!(sanitize_name("sub/dir/x"), Some("subdirx.safetensors".into()));
        assert_eq!(sanitize_name(".hidden"), None);
        assert_eq!(sanitize_name("  "), None);
    }

    #[test]
    fn net_config_metadata_round_trip_and_detection() {
        use crate::games::ultimate_ttt::{self, UltimateTicTacToe};
        use crate::training::AlphaZeroTrainer;
        const G: usize = ultimate_ttt::INPUT_SIZE;
        let dir = std::env::temp_dir().join(format!("mcts_ckpt_meta_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mlp = ultimate_ttt::net_mlp();
        let trainer = AlphaZeroTrainer::<G, 81, 2>::new(mlp, 1e-3, 1e-4, crate::backend::NnDevice::CPU).unwrap();

        // Saving records the architecture; loading the weights back still works.
        let saved = dir.join("uttt_meta.safetensors");
        trainer.save_checkpoint(&saved).unwrap();
        assert_eq!(read_net_config(&saved), Some(mlp));
        let mut other = AlphaZeroTrainer::<G, 81, 2>::new(mlp, 1e-3, 1e-4, crate::backend::NnDevice::CPU).unwrap();
        other.load_checkpoint(&saved).unwrap();

        // A file without metadata (as written before) is recognised by its tensors.
        let varmap = VarMap::new();
        AlphaZeroNet::<G, 81, 2>::new(VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu), &mlp).unwrap();
        let bare = dir.join("uttt_bare.safetensors");
        varmap.save(&bare).unwrap();
        assert_eq!(read_net_config(&bare), None);
        assert_eq!(net_config::<UltimateTicTacToe, G, 81, 2>(&bare), Ok(mlp));
        assert_eq!(preset_name::<UltimateTicTacToe, G, 81, 2>(&mlp), "mlp");

        // Metadata written before `per_block_context` existed reads as the shared context.
        let old_json = r#"{"kind":"block_net","in_planes":7,"board_height":9,"board_width":9,"block_height":3,"block_width":3,"local_dim":64,"hidden_dim":256,"num_res_blocks":10,"head_hidden":64}"#;
        let old: AlphaZeroNetConfig = serde_json::from_str(old_json).unwrap();
        assert_eq!(old, ultimate_ttt::net_block_deeper());
        assert_eq!(preset_name::<UltimateTicTacToe, G, 81, 2>(&old), "block_deeper");
        assert_eq!(preset_name::<UltimateTicTacToe, G, 81, 2>(&ultimate_ttt::net_block_ctx()), "block_ctx");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
