//! Append-only JSON-lines files: the eval log and the round dumps.

use std::io::Write;
use std::path::Path;

/// Appends `value` as one line, creating the file if needed.
pub(super) fn append_jsonl(path: &Path, value: &impl serde::Serialize) -> std::io::Result<()> {
    let line = serde_json::to_string(value).map_err(std::io::Error::other)?;
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(file, "{line}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Serialize)]
    struct Row {
        name: &'static str,
        n: u32,
    }

    #[test]
    fn append_jsonl_appends_one_line_per_call() {
        let dir = std::env::temp_dir().join(format!("jsonl-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rows.jsonl");
        append_jsonl(&path, &serde_json::json!({"a": 1})).unwrap();
        let row = Row { name: "x", n: 2 };
        append_jsonl(&path, &row).unwrap();
        let expected = format!("{{\"a\":1}}\n{}\n", serde_json::to_string(&row).unwrap());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), expected);
        std::fs::remove_dir_all(dir).ok();
    }
}
