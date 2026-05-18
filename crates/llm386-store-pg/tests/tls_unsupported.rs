//! Verifies that opening with a TLS mode that requires the
//! `tls-native-tls` feature fails *loudly* with
//! `StoreOpenError::TlsUnsupported` when that feature isn't enabled.
//!
//! This file always compiles. The actual assertion only runs when the
//! `tls-native-tls` feature is OFF — when it's on, both modes work
//! and the assertion-of-failure would be wrong, so the test no-ops.

use llm386_store_pg::{PgStore, PgStoreConfig, TlsMode};

#[test]
fn tls_modes_error_when_feature_disabled() {
    // The URL never gets parsed — the TLS check fires first.
    let url = "postgres://invalid/never_used";

    let result = PgStore::open(
        url,
        &PgStoreConfig {
            tls: TlsMode::Require,
            ..Default::default()
        },
    );

    #[cfg(not(feature = "tls-native-tls"))]
    {
        match result {
            Err(llm386_store_pg::StoreOpenError::TlsUnsupported(_)) => {}
            Err(other) => panic!(
                "expected StoreOpenError::TlsUnsupported, got: {other:?}"
            ),
            Ok(_) => panic!("expected error, got Ok"),
        }
    }
    #[cfg(feature = "tls-native-tls")]
    {
        // With the feature on, `Require` is valid and the open
        // proceeds to actually try to connect — which fails with a
        // `Connect` error (because the URL is bogus), not
        // `TlsUnsupported`. That's the right behavior; skip the
        // detailed assertion.
        let _ = result;
    }
}
