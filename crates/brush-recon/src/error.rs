use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum ReconError {
    #[error("invalid reconstruction input: {0}")]
    InvalidInput(String),
    #[error("only pinhole cameras are supported by this experiment")]
    UnsupportedCamera,
    #[error("backend does not support requested output: {0}")]
    UnsupportedOutput(&'static str),
    #[error("renderer needs {requested} bytes, budget is {budget}")]
    BufferBudget { requested: u64, budget: u64 },
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path}: {source}")]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error("GPU error: {0}")]
    Gpu(String),
    #[error("invalid model state: {0}")]
    ModelState(String),
}

impl ReconError {
    pub fn input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            source,
        }
    }
}
