//! The `/v1/messages` client.
//!
//! Named `api` rather than `client` on purpose. `client` in this workspace is
//! the memory daemon's socket client, and this crate exists partly to show
//! what linking the engine directly looks like instead.
//!
//! Everything here except `Api::send` is a pure function over bytes, which is
//! what lets the whole decoder be tested in a sandbox with no network and no
//! API key -- the environment `nix build` runs the test suite in.

mod sse;
