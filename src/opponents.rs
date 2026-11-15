//! Configurable evaluation opponents for the Browser Lab (live-training arena and experiments).

use serde::{Deserialize, Serialize};

/// One evaluation opponent and its parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OpponentSpec {
    /// Uniformly random legal moves.
    Random,
    /// Classical UCT MCTS with random playouts.
    Mcts {
        sims: usize,
        #[serde(default = "default_uct_c")]
        exploration: f32,
    },
    /// Alpha-beta Minimax with the game's heuristic.
    Minimax { depth: u32 },
    /// A saved network (`checkpoint` file, architecture preset `net`) searching `sims` per move in
    /// each of `trees` root-parallel trees, whose visit counts are summed to choose the move. More
    /// trees search more without taking longer on a GPU: their positions are evaluated as one batch.
    #[serde(rename = "alphazero")]
    AlphaZero {
        checkpoint: String,
        net: String,
        sims: usize,
        #[serde(default = "one_tree")]
        trees: usize,
    },
}

const fn default_uct_c() -> f32 {
    1.414
}

const fn one_tree() -> usize {
    1
}

impl OpponentSpec {
    pub fn label(&self) -> String {
        match self {
            Self::Random => "Random".to_string(),
            Self::Mcts { sims, .. } => format!("MCTS-{sims} rollouts"),
            Self::Minimax { depth } => format!("Minimax d={depth}"),
            Self::AlphaZero { checkpoint, sims, trees, .. } => {
                let trees = if *trees > 1 { format!("×{trees}") } else { String::new() };
                format!("AZ {} @{sims}{trees}", checkpoint.trim_end_matches(".safetensors"))
            }
        }
    }

    /// Clamps parameters to sane ranges.
    pub fn sanitized(self) -> Self {
        match self {
            Self::Mcts { sims, exploration } => Self::Mcts { sims: sims.clamp(1, 1_000_000), exploration: exploration.clamp(0.0, 10.0) },
            Self::Minimax { depth } => Self::Minimax { depth: depth.clamp(1, 12) },
            Self::AlphaZero { checkpoint, net, sims, trees } => {
                Self::AlphaZero { checkpoint, net, sims: sims.clamp(1, 100_000), trees: trees.clamp(1, 64) }
            }
            other => other,
        }
    }
}

/// Parses a JSON array of opponents (invalid entries are skipped).
pub fn parse_list(v: &serde_json::Value) -> Option<Vec<OpponentSpec>> {
    let arr = v.as_array()?;
    Some(
        arr.iter()
            .filter_map(|o| serde_json::from_value::<OpponentSpec>(o.clone()).ok())
            .map(OpponentSpec::sanitized)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip_and_labels() {
        let list = vec![
            OpponentSpec::Random,
            OpponentSpec::Mcts { sims: 200, exploration: 1.414 },
            OpponentSpec::Minimax { depth: 2 },
            OpponentSpec::AlphaZero { checkpoint: "uttt_exp4_block.safetensors".into(), net: "block".into(), sims: 100, trees: 8 },
        ];
        let json = serde_json::to_value(&list).unwrap();
        assert_eq!(json[1]["kind"], "mcts");
        assert_eq!(json[3]["kind"], "alphazero");
        assert_eq!(parse_list(&json).unwrap(), list);
        assert_eq!(list[3].label(), "AZ uttt_exp4_block @100×8");
        // Missing optional parameter falls back to its default.
        let m: OpponentSpec = serde_json::from_str(r#"{"kind":"mcts","sims":50}"#).unwrap();
        assert_eq!(m, OpponentSpec::Mcts { sims: 50, exploration: 1.414 });
        let az: OpponentSpec = serde_json::from_str(r#"{"kind":"alphazero","checkpoint":"a.safetensors","net":"block","sims":50}"#).unwrap();
        assert!(matches!(az, OpponentSpec::AlphaZero { trees: 1, .. }), "saved specs without trees use one tree");
    }
}
