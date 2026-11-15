//! The folder that holds `.safetensors` checkpoints.
//!
//! The Browser Lab and its HTTP API name checkpoints by plain file name
//! (`uttt_exp4_block.safetensors`); this module maps those names to one folder: `checkpoints/`
//! under the working directory, or `$MCTS_CHECKPOINT_DIR` (e.g. a mounted volume in a container).

use std::path::PathBuf;

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
}
