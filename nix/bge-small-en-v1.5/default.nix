{ runCommand, fetchurl }:
let
  rev = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a";
  base = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/${rev}";
  model = fetchurl {
    url = "${base}/onnx/model.onnx";
    hash = "sha256-go4Ultf6u3nPpNzYT6OGJcDT0h2kdKAPCNsPVZlAzzU=";
  };
  tokenizer = fetchurl {
    url = "${base}/tokenizer.json";
    hash = "sha256-0kGmDV6PBMwbKz6e96SSGye/Um2fYFCrkPkmeh+eXGY=";
  };
in
# The embedder joins model_dir with exactly these two names and nothing else,
# so this layout is the whole contract. Revision is pinned rather than `main`:
# a moving ref would silently change the embeddings under a warm index and
# nothing would report it.
runCommand "bge-small-en-v1.5" { } ''
  mkdir -p $out
  cp ${model} $out/model.onnx
  cp ${tokenizer} $out/tokenizer.json
''
