use std::process::{Child, Command, Stdio};

pub const SERVICE_TYPE: &str = "_brushguide._tcp";

/// Arguments for `dns-sd -R <instance> <type> <domain> <port> <txt records...>`,
/// registering `brush-guide on <hostname>` under `SERVICE_TYPE` with TXT `proto=1`.
pub fn register_args(port: u16, hostname: &str) -> Vec<String> {
    vec![
        "-R".to_owned(),
        format!("brush-guide on {hostname}"),
        SERVICE_TYPE.to_owned(),
        "local".to_owned(),
        port.to_string(),
        "proto=1".to_owned(),
    ]
}

/// A running `dns-sd -R` registration; dropping it unregisters the service.
pub struct Advertisement(Child);

impl Drop for Advertisement {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Advertise the server on the local network via the system's mDNSResponder;
/// drop the returned `Advertisement` to stop.
pub fn advertise(port: u16) -> anyhow::Result<Advertisement> {
    let hostname = hostname::get()?.to_string_lossy().into_owned();
    let child = Command::new("/usr/bin/dns-sd")
        .args(register_args(port, &hostname))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    Ok(Advertisement(child))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_args_has_type_port_and_txt() {
        let args = register_args(8765, "studio");
        assert_eq!(
            args,
            vec![
                "-R",
                "brush-guide on studio",
                "_brushguide._tcp",
                "local",
                "8765",
                "proto=1",
            ]
        );
    }
}
