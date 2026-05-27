//! TLS connector construction for both [`crate::PgStore`] and
//! [`crate::AsyncPgStore`].
//!
//! The same `native-tls` connector is consumed by sync `postgres`
//! (via `postgres-native-tls`) and async `tokio-postgres` (via the
//! same crate — its `MakeTlsConnector` implements the
//! `tokio_postgres::tls::MakeTlsConnect` trait that both sides use).
//! So one feature gate covers both backends.

use crate::common::{StoreOpenError, TlsMode};

/// A type-erased TLS configuration handle. With the
/// `tls-native-tls` feature enabled, [`TlsMode::Require`] /
/// [`TlsMode::RequireCustomCa`] produce a [`Self::Native`] variant
/// wrapping a `native_tls::TlsConnector` that's compatible with both
/// the sync and async Postgres clients.
#[cfg_attr(not(feature = "tls-native-tls"), allow(dead_code))]
pub(crate) enum TlsBackend {
    Plain,
    #[cfg(feature = "tls-native-tls")]
    Native(native_tls::TlsConnector),
}

#[cfg_attr(not(feature = "tls-native-tls"), allow(unused_variables))]
pub(crate) fn build_backend(mode: &TlsMode) -> Result<TlsBackend, StoreOpenError> {
    match mode {
        TlsMode::Disable => Ok(TlsBackend::Plain),
        #[cfg(feature = "tls-native-tls")]
        TlsMode::Require => {
            let connector = native_tls::TlsConnector::new()
                .map_err(|e| StoreOpenError::TlsConfig(format!("native-tls init: {e}")))?;
            Ok(TlsBackend::Native(connector))
        }
        #[cfg(feature = "tls-native-tls")]
        TlsMode::RequireNoverify => {
            let connector = native_tls::TlsConnector::builder()
                .danger_accept_invalid_certs(true)
                .danger_accept_invalid_hostnames(true)
                .build()
                .map_err(|e| StoreOpenError::TlsConfig(format!("native-tls build: {e}")))?;
            Ok(TlsBackend::Native(connector))
        }
        #[cfg(feature = "tls-native-tls")]
        TlsMode::RequireCustomCa { ca_path } => {
            let pem = std::fs::read(ca_path).map_err(|e| {
                StoreOpenError::TlsConfig(format!(
                    "read CA bundle at {}: {e}",
                    ca_path.display()
                ))
            })?;
            let cert = native_tls::Certificate::from_pem(&pem)
                .map_err(|e| StoreOpenError::TlsConfig(format!("parse CA bundle: {e}")))?;
            let connector = native_tls::TlsConnector::builder()
                .add_root_certificate(cert)
                .build()
                .map_err(|e| StoreOpenError::TlsConfig(format!("native-tls build: {e}")))?;
            Ok(TlsBackend::Native(connector))
        }
        #[cfg(not(feature = "tls-native-tls"))]
        other => Err(StoreOpenError::TlsUnsupported(other.clone())),
    }
}
