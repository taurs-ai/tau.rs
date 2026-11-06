//! HTTP server configuration.

use std::net::SocketAddr;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HttpServerConfig {
    listen_addr: SocketAddr,
    max_requests: Option<usize>,
}

impl HttpServerConfig {
    /// Creates HTTP server configuration for a listen address.
    ///
    /// The input address is the socket to bind. The output has no request limit by
    /// default and cannot fail.
    pub fn new(listen_addr: SocketAddr) -> Self {
        Self {
            listen_addr,
            max_requests: None,
        }
    }

    /// Sets an optional maximum number of requests before the server exits.
    ///
    /// The input value is `Some(count)` for bounded serving or `None` for normal
    /// continuous serving. The function returns the updated config and performs no
    /// validation.
    pub fn with_max_requests(mut self, max_requests: Option<usize>) -> Self {
        self.max_requests = max_requests;
        self
    }

    /// Returns the configured listen address.
    ///
    /// The method borrows the config and returns the stored socket address by
    /// value. It cannot fail.
    pub fn listen_addr(&self) -> SocketAddr {
        self.listen_addr
    }

    /// Returns the optional maximum request count.
    ///
    /// The method borrows the config and returns `Some(count)` for bounded serving
    /// or `None` for continuous serving. It cannot fail.
    pub(crate) fn max_requests(&self) -> Option<usize> {
        self.max_requests
    }
}
