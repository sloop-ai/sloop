//! BGE-small-en-v1.5 inference over onnxruntime.
//!
//! Weights are pinned into the nix store and located via `SLOOP_MEMORY_MODEL`, and
//! `ort` is built with `load-dynamic`, so nothing here ever touches the network.

use anyhow::{anyhow, Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

use crate::config;

/// Embedding is ~96% of index wall time, so thread count is the main lever short
/// of a different execution provider. Capped rather than set to the full core
/// count because efficiency cores contribute little here and oversubscription
/// costs more than it gains.
fn intra_threads() -> usize {
    std::env::var("SLOOP_MEMORY_EMBED_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get().min(8)))
}

pub struct Embedder {
    tokenizer: Tokenizer,
    session: Session,
    input_names: Vec<String>,
}

impl Embedder {
    /// # Errors
    ///
    /// Returns an error if the tokenizer or ONNX model under `model_dir`
    /// cannot be loaded, or the session cannot be configured.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("loading tokenizer: {e}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config::MAX_SEQ_LEN,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|e| anyhow!("configuring truncation: {e}"))?;
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));

        let model_path = model_dir.join("model.onnx");
        // ort's builder errors hand the builder back inside the error, which makes
        // them !Send and so not convertible into anyhow::Error by `?`. Flatten to
        // a message at each step instead.
        let session = Session::builder()
            .map_err(|e| anyhow!("creating ort session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("setting graph optimization level: {e}"))?
            .with_intra_threads(intra_threads())
            .map_err(|e| anyhow!("setting intra-op threads: {e}"))?
            .commit_from_file(&model_path)
            .with_context(|| format!("opening ONNX model at {}", model_path.display()))?;

        let input_names = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();

        Ok(Self {
            tokenizer,
            session,
            input_names,
        })
    }

    fn has_input(&self, name: &str) -> bool {
        self.input_names.iter().any(|n| n == name)
    }

    fn forward(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("tokenizing: {e}"))?;

        let batch = encodings.len();
        let seq = encodings.first().map_or(0, |e| e.get_ids().len());
        if batch == 0 || seq == 0 {
            return Ok(vec![]);
        }
        // batch is bounded by the caller's input list and seq by the tokenizer's
        // configured max length; neither can approach i64::MAX (9.2 quintillion).
        #[expect(clippy::cast_possible_wrap, reason = "see comment above")]
        let shape = vec![batch as i64, seq as i64];

        let ids: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_ids().iter().map(|v| i64::from(*v)))
            .collect();
        let mask: Vec<i64> = encodings
            .iter()
            .flat_map(|e| e.get_attention_mask().iter().map(|v| i64::from(*v)))
            .collect();

        let mut inputs: Vec<(&str, ort::value::DynValue)> = vec![
            (
                "input_ids",
                Tensor::from_array((shape.clone(), ids))?.into_dyn(),
            ),
            (
                "attention_mask",
                Tensor::from_array((shape.clone(), mask))?.into_dyn(),
            ),
        ];
        // Some exports of this model omit token_type_ids; only feed declared inputs.
        if self.has_input("token_type_ids") {
            let types: Vec<i64> = encodings
                .iter()
                .flat_map(|e| e.get_type_ids().iter().map(|v| i64::from(*v)))
                .collect();
            inputs.push((
                "token_type_ids",
                Tensor::from_array((shape, types))?.into_dyn(),
            ));
        }
        inputs.retain(|(name, _)| self.has_input(name));

        let outputs = self.session.run(inputs)?;
        let (out_shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        // This feeds the slice bounds below; a corrupted or unexpected model
        // export producing a negative or oversized dimension must fail loudly
        // rather than wrap into a bogus slice length.
        let hidden = usize::try_from(
            *out_shape
                .last()
                .context("model output has no trailing dimension")?,
        )
        .context("model output has a negative trailing dimension")?;

        // BGE pools the CLS token rather than mean-pooling. Getting this wrong
        // degrades retrieval quietly, which is the worst way for it to break.
        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let start = b * seq * hidden;
            let cls = &data[start..start + hidden];
            let norm = cls.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            out.push(cls.iter().map(|v| v / norm).collect());
        }
        Ok(out)
    }

    pub(crate) fn encode(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        const BATCH: usize = 32;
        let mut out = Vec::with_capacity(texts.len());
        for window in texts.chunks(BATCH) {
            out.extend(self.forward(window)?);
        }
        Ok(out)
    }

    /// # Errors
    ///
    /// Returns an error if encoding fails, or if the model produces no rows
    /// for the (single-element) query batch.
    pub fn encode_query(&mut self, query: &str) -> Result<Vec<f32>> {
        let prefixed = format!("{}{}", config::QUERY_PREFIX, query);
        self.encode(std::slice::from_ref(&prefixed))?
            .pop()
            .context("embedder returned no rows for query")
    }
}
