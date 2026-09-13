//! Hybrid local search engine over private markdown roots.
//!
//! This library exists so consumers link the search engine directly instead
//! of talking to it over a socket: `sloop-memory` embeds it for the resident
//! daemon, MCP server, and prompt-hook client, and future consumers can do
//! the same without spawning a subprocess.

pub mod chunk;
pub mod client;
pub mod config;
pub mod embed;
pub mod index;
pub mod proto;
pub mod store;
