//! Offline adapters; intentionally independent of GuideSession and the phone protocol.
pub mod cli;
pub mod gaussian;
pub mod io;
pub mod metrics;
pub mod preflight;
pub mod report;
pub mod run;
pub mod runtime;
pub mod triangle;

pub(crate) fn training_ids(
    train: &[std::sync::Arc<brush_recon::ReconFrame>],
) -> Result<std::collections::BTreeSet<u64>, brush_recon::ReconError> {
    if train.is_empty() {
        return Err(brush_recon::ReconError::input(
            "initialization requires training frames",
        ));
    }
    let mut ids = std::collections::BTreeSet::new();
    for frame in train {
        frame.validate()?;
        if !ids.insert(frame.id) {
            return Err(brush_recon::ReconError::input(
                "duplicate training frame ID",
            ));
        }
    }
    Ok(ids)
}
