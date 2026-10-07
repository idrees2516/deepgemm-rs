//! Error types for DeepGEMM-RS.

use std::fmt;

#[derive(Debug)]
pub enum DgError {
    Driver(String),
    Nvrtc(String),
    InvalidArg(String),
    Unsupported(String),
    Mismatch(String),
    Io(std::io::Error),
}

impl fmt::Display for DgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DgError::Driver(s) => write!(f, "CUDA driver error: {s}"),
            DgError::Nvrtc(s) => write!(f, "NVRTC compile error: {s}"),
            DgError::InvalidArg(s) => write!(f, "invalid argument: {s}"),
            DgError::Unsupported(s) => write!(f, "unsupported operation: {s}"),
            DgError::Mismatch(s) => write!(f, "data mismatch: {s}"),
            DgError::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for DgError {}

impl From<std::io::Error> for DgError {
    fn from(e: std::io::Error) -> Self {
        DgError::Io(e)
    }
}

pub type DgResult<T> = Result<T, DgError>;
