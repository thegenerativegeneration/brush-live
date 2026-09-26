use mdns_sd::{ServiceDaemon, ServiceInfo};
use std::collections::HashMap;

pub const SERVICE_TYPE: &str = "_brushguide._tcp.local.";

pub fn service_info(port: u16, hostname: &str) -> anyhow::Result<ServiceInfo> {
    let host = format!("{}.local.", hostname.trim_end_matches(".local"));
    let props = HashMap::from([("proto".to_owned(), "1".to_owned())]);
    let info = ServiceInfo::new(
        SERVICE_TYPE,
        &format!("brush-guide on {hostname}"),
        &host,
        "",
        port,
        props,
    )?
    .enable_addr_auto();
    Ok(info)
}

/// Advertise the server on the local network; drop the daemon to stop.
pub fn advertise(port: u16) -> anyhow::Result<ServiceDaemon> {
    let hostname = hostname::get()?.to_string_lossy().into_owned();
    let daemon = ServiceDaemon::new()?;
    daemon.register(service_info(port, &hostname)?)?;
    Ok(daemon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_info_has_type_port_and_txt() {
        let info = service_info(8765, "studio").unwrap();
        assert_eq!(info.get_type(), SERVICE_TYPE);
        assert_eq!(info.get_port(), 8765);
        assert_eq!(
            info.get_fullname(),
            "brush-guide on studio._brushguide._tcp.local."
        );
        assert_eq!(info.get_property_val_str("proto"), Some("1"));
    }
}
