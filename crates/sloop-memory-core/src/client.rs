//! Synchronous daemon client.
//!
//! Deliberately not async: the hook runs on every prompt the user types, and
//! standing up a tokio runtime to write one line to a socket would cost more than
//! the query itself.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

use anyhow::{Context, Result};

use crate::proto::{Request, Response};

/// # Errors
///
/// Returns an error if the daemon socket cannot be connected to, the request
/// cannot be written or the connection times out, or the response is not
/// valid JSON.
pub fn request(req: &Request, timeout: Duration) -> Result<Response> {
    let path = crate::config::socket_path();
    let stream = UnixStream::connect(&path)
        .with_context(|| format!("connecting to daemon at {}", path.display()))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;

    let mut writer = stream.try_clone()?;
    writer.write_all(serde_json::to_string(req)?.as_bytes())?;
    writer.write_all(b"\n")?;
    writer.flush()?;

    let mut line = String::new();
    BufReader::new(stream).read_line(&mut line)?;
    serde_json::from_str(&line).context("parsing daemon response")
}
