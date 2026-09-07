{
  description = "sloop: memory for AI coding harnesses";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { nixpkgs, rust-overlay, ... }:
    let
      # x86_64-darwin is absent because nixpkgs-unstable dropped it: Intel Macs
      # are unsupported there as of 26.11.
      systems = [ "aarch64-darwin" "aarch64-linux" "x86_64-linux" ];

      forAllSystems = f: nixpkgs.lib.genAttrs systems (system:
        f (import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        }));

      # cargo, rustc, clippy and rustfmt from one release, pinned in
      # rust-toolchain.toml. Keeping them to a single source is the whole point
      # -- see the comment in that file for what mixing them costs. The dev
      # shell and the package build share this, so what CI checks is what a
      # contributor runs.
      toolchainFor = pkgs: pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

      # ort uses load-dynamic, so onnxruntime is dlopened at runtime rather than
      # linked. The extension resolves to .dylib on darwin and .so on linux.
      ortPathFor = pkgs: "${pkgs.onnxruntime}/lib/libonnxruntime"
        + pkgs.stdenv.hostPlatform.extensions.sharedLibrary;

      modelFor = pkgs: pkgs.callPackage ./nix/bge-small-en-v1.5 { };
    in
    {
      packages = forAllSystems (pkgs:
        let
          toolchain = toolchainFor pkgs;
          model = modelFor pkgs;
          ortPath = ortPathFor pkgs;

          # Build with the pinned toolchain rather than nixpkgs' rustc, so the
          # package and the dev shell compile with the same compiler.
          rustPlatform = pkgs.makeRustPlatform {
            cargo = toolchain;
            rustc = toolchain;
          };

          sloop-memory = rustPlatform.buildRustPackage {
            pname = "sloop-memory";
            version = "0.1.0";

            # Only the files cargo actually reads. Taking ./. wholesale -- or
            # even ./crates wholesale, which sweeps in the crate READMEs --
            # rebuilds the entire dependency tree whenever a doc changes. No
            # crate pulls a README in with include_str!, so dropping them is
            # safe. rust-toolchain.toml is excluded too: the compiler comes
            # from makeRustPlatform, and cargo would only read that file
            # through a rustup shim that does not exist in the sandbox.
            src = nixpkgs.lib.fileset.toSource {
              root = ./.;
              fileset = nixpkgs.lib.fileset.unions [
                ./Cargo.toml
                ./Cargo.lock
                (nixpkgs.lib.fileset.fileFilter
                  (f: f.hasExt "rs" || f.name == "Cargo.toml")
                  ./crates)
              ];
            };

            # cargoHash with fetchCargoVendor, NOT cargoLock.lockFile.
            # nixpkgs' importCargoLock fetches crates from crates.io's /api/v1
            # endpoint, which crates.io rate-limits. Cargo's own vendoring uses
            # the sparse registry plus the static.crates.io CDN, which is not.
            # Update this hash whenever Cargo.lock changes.
            cargoHash = "sha256-QrI+87TMnXW0gTPB3CAxh5C7kF0upfTNg85kqL8jC5s=";
            cargoBuildFlags = [ "-p" "sloop-memory" ];

            # protoc is build-time codegen for lance-encoding's .proto files,
            # not something we link against.
            nativeBuildInputs = [ pkgs.protobuf pkgs.makeWrapper ];

            # The tests need the same two runtime inputs the binary does.
            ORT_DYLIB_PATH = ortPath;
            SLOOP_MEMORY_MODEL = model;
            cargoTestFlags = [ "--workspace" ];

            # Bake in the two inputs the binary cannot start without, so an
            # installed sloop-memory works without the caller reproducing the
            # dev shell. SLOOP_MEMORY_ROOTS is deliberately not set here: it is
            # the one value only the operator can supply.
            postInstall = ''
              wrapProgram $out/bin/sloop-memory \
                --set-default ORT_DYLIB_PATH ${ortPath} \
                --set-default SLOOP_MEMORY_MODEL ${model}
            '';

            meta = {
              description = "Hybrid local search over private markdown roots";
              homepage = "https://github.com/sloop-ai/sloop";
              mainProgram = "sloop-memory";
              license = nixpkgs.lib.licenses.mit;
            };
          };
        in
        {
          inherit sloop-memory;
          default = sloop-memory;
        });

      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          nativeBuildInputs = [
            (toolchainFor pkgs)
            pkgs.protobuf
          ];

          ORT_DYLIB_PATH = ortPathFor pkgs;

          # The library refuses to start without a model path, by design.
          SLOOP_MEMORY_MODEL = modelFor pkgs;
        };
      });
    };
}
