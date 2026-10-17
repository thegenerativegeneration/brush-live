use brush_recon::{ReconError, capture::AssetReference};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    fs::OpenOptions,
    io::Write,
    path::{Component, Path, PathBuf},
};

pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn read(path: &Path) -> Result<Vec<u8>, ReconError> {
    std::fs::read(path).map_err(|e| ReconError::io(path, e))
}
pub fn json<T: Serialize>(path: &Path, value: &T) -> Result<(), ReconError> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|e| ReconError::input(format!("{}: {e}", path.display())))?;
    write_new(path, &bytes)
}
pub fn write_new(path: &Path, bytes: &[u8]) -> Result<(), ReconError> {
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| ReconError::io(path, e))?;
    f.write_all(bytes).map_err(|e| ReconError::io(path, e))
}
pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, ReconError> {
    serde_json::from_slice(&read(path)?).map_err(|source| ReconError::Json {
        path: path.to_path_buf(),
        source,
    })
}
pub fn validate_output(source: &Path, output: &Path) -> Result<PathBuf, ReconError> {
    let source = source
        .canonicalize()
        .map_err(|e| ReconError::io(source, e))?;
    let boundary = source
        .parent()
        .filter(|p| p.join("capture.json").is_file())
        .unwrap_or(&source);
    if output
        .components()
        .any(|c| matches!(c, Component::ParentDir))
    {
        return Err(ReconError::input("output must not contain '..' components"));
    }
    if output.symlink_metadata().is_ok() {
        return Err(ReconError::input(format!(
            "output already exists: {}",
            output.display()
        )));
    }
    let absolute = if output.is_absolute() {
        output.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| ReconError::io(".", e))?
            .join(output)
    };
    let mut ancestor = absolute.as_path();
    let mut missing = vec![];
    while !ancestor
        .try_exists()
        .map_err(|e| ReconError::io(ancestor, e))?
    {
        if ancestor.symlink_metadata().is_ok() {
            return Err(ReconError::input("output ancestry has a dangling symlink"));
        }
        missing.push(
            ancestor
                .file_name()
                .ok_or_else(|| ReconError::input("output has no filename"))?
                .to_os_string(),
        );
        ancestor = ancestor
            .parent()
            .ok_or_else(|| ReconError::input("output has no existing ancestor"))?;
    }
    let mut resolved = ancestor
        .canonicalize()
        .map_err(|e| ReconError::io(ancestor, e))?;
    for part in missing.into_iter().rev() {
        resolved.push(part);
    }
    if resolved.starts_with(boundary) || boundary.starts_with(&resolved) {
        return Err(ReconError::input(
            "output must be outside the capture/source tree",
        ));
    }
    Ok(resolved)
}
pub fn create_output(path: &Path) -> Result<(), ReconError> {
    let parent = path
        .parent()
        .ok_or_else(|| ReconError::input("output needs a parent"))?;
    std::fs::create_dir_all(parent).map_err(|e| ReconError::io(parent, e))?;
    std::fs::create_dir(path).map_err(|e| ReconError::io(path, e))
}
pub fn asset(root: &Path, reference: &AssetReference) -> Result<Vec<u8>, ReconError> {
    if reference.file.is_absolute()
        || reference
            .file
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(ReconError::input("unsafe frozen asset path"));
    }
    let path = root.join(&reference.file);
    let canonical = path.canonicalize().map_err(|e| ReconError::io(&path, e))?;
    let root = root.canonicalize().map_err(|e| ReconError::io(root, e))?;
    if !canonical.starts_with(&root) {
        return Err(ReconError::input("frozen asset escapes its directory"));
    }
    let bytes = read(&path)?;
    if digest(&bytes) != reference.sha256 {
        return Err(ReconError::input(format!(
            "frozen asset changed: {}",
            path.display()
        )));
    }
    Ok(bytes)
}

pub fn provenance() -> Result<(String, String), ReconError> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let git = |args: &[&str]| -> Result<Vec<u8>, ReconError> {
        let out = std::process::Command::new("git")
            .current_dir(&root)
            .args(args)
            .output()
            .map_err(|e| ReconError::io(&root, e))?;
        if !out.status.success() {
            return Err(ReconError::input(format!(
                "git provenance: {}",
                String::from_utf8_lossy(&out.stderr)
            )));
        }
        Ok(out.stdout)
    };
    let revision = String::from_utf8_lossy(&git(&["rev-parse", "HEAD"])?)
        .trim()
        .to_string();
    let mut dirty = git(&["diff", "--binary", "HEAD"])?;
    let names = git(&["ls-files", "--others", "--exclude-standard", "-z"])?;
    for name in names.split(|b| *b == 0).filter(|n| !n.is_empty()) {
        let name = std::str::from_utf8(name)
            .map_err(|_| ReconError::input("non-UTF8 untracked path in worktree provenance"))?;
        dirty.extend_from_slice(name.as_bytes());
        dirty.push(0);
        dirty.extend_from_slice(digest(&read(&root.join(name))?).as_bytes());
        dirty.push(0);
    }
    Ok((revision, digest(&dirty)))
}
