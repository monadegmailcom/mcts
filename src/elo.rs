//! Anchored Elo ratings for the Browser Lab.
//!
//! An Elo difference is a match score in other units: a player rated `D` points higher is expected
//! to score `1 / (1 + 10^(-D/400))` per game (wins + ½ draws). Scores alone are relative to one
//! opponent and saturate near 0 % and 100 %, so the lab rates networks against a fixed pool of
//! *anchors* (Random, Minimax, MCTS, saved networks):
//!
//! 1. The anchors play a round-robin once; a Bradley-Terry fit of all pairwise results gives each
//!    anchor a rating, the first anchor fixed at 0 ([`EloPool::calibrate`]).
//! 2. A network's rating is the one that best explains its results against whichever anchors it
//!    was evaluated against ([`EloPool::estimate`]). No extra games are needed, and wins against
//!    weak anchors and close games against strong ones all count, so the scale does not saturate.
//!
//! Every pairing also counts one virtual draw, a weak prior that keeps ratings finite when a player
//! wins or loses every game. Standard errors treat each game as a win or loss, so with draws they
//! are slightly pessimistic.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::opponents::OpponentSpec;

/// Elo points per natural-log unit of the odds (the fit works in log-odds).
const ELO_PER_NAT: f64 = 400.0 / std::f64::consts::LN_10;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rating {
    pub elo: f64,
    /// Standard error of `elo`.
    pub stderr: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Anchor {
    pub opponent: OpponentSpec,
    pub rating: Rating,
}

/// The games between anchors `a` and `b`: `[wins, draws, losses]` from `a`'s point of view.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pairing {
    pub a: usize,
    pub b: usize,
    pub wdl: [usize; 3],
}

/// Calibrated anchors of one game, with the round-robin they were fitted to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EloPool {
    pub anchors: Vec<Anchor>,
    pub pairings: Vec<Pairing>,
    pub games_per_pair: usize,
    pub calibrated_unix: u64,
}

impl EloPool {
    /// Fits ratings for `opponents` to the round-robin `pairings` (indices into `opponents`); the
    /// first opponent is fixed at 0 Elo.
    pub fn calibrate(opponents: Vec<OpponentSpec>, pairings: Vec<Pairing>, games_per_pair: usize) -> Self {
        let n = opponents.len();
        let mut x = vec![0.0f64; n];
        let mut info = vec![0.0f64; n];
        // Coordinate-wise Newton on the concave log-likelihood; converges in a few dozen sweeps.
        for _ in 0..1000 {
            let mut largest_step = 0.0f64;
            for i in 1..n {
                let games = pairings.iter().filter_map(|p| {
                    if p.a == i {
                        Some(Games::new(x[p.b], p.wdl))
                    } else if p.b == i {
                        Some(Games::new(x[p.a], [p.wdl[2], p.wdl[1], p.wdl[0]]))
                    } else {
                        None
                    }
                });
                let (step, fisher) = newton_step(x[i], games);
                x[i] += step;
                info[i] = fisher;
                largest_step = largest_step.max(step.abs());
            }
            if largest_step < 1e-9 {
                break;
            }
        }
        let anchors = opponents
            .into_iter()
            .zip(x.iter().zip(&info))
            .enumerate()
            .map(|(i, (opponent, (&x, &info)))| Anchor {
                opponent,
                rating: Rating { elo: x * ELO_PER_NAT, stderr: if i == 0 { 0.0 } else { stderr(info) } },
            })
            .collect();
        Self { anchors, pairings, games_per_pair, calibrated_unix: unix_now() }
    }

    /// The rating that best explains `results` (`[wins, draws, losses]` against each opponent).
    /// Opponents that are not anchors, or without games, are ignored; `None` if nothing is left.
    pub fn estimate<'a>(&self, results: impl IntoIterator<Item = (&'a OpponentSpec, [usize; 3])>) -> Option<Rating> {
        let games: Vec<Games> = results
            .into_iter()
            .filter(|(_, wdl)| wdl.iter().sum::<usize>() > 0)
            .filter_map(|(opp, wdl)| {
                let anchor = self.anchors.iter().find(|a| a.opponent == *opp)?;
                Some(Games::new(anchor.rating.elo / ELO_PER_NAT, wdl))
            })
            .collect();
        if games.is_empty() {
            return None;
        }
        let mut x = games.iter().map(|g| g.opponent).sum::<f64>() / games.len() as f64;
        let mut info = 0.0;
        for _ in 0..100 {
            let (step, fisher) = newton_step(x, games.iter().copied());
            x += step;
            info = fisher;
            if step.abs() < 1e-9 {
                break;
            }
        }
        Some(Rating { elo: x * ELO_PER_NAT, stderr: stderr(info) })
    }

    pub fn load(path: &Path) -> Option<Self> {
        serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        std::fs::write(path, serde_json::to_string_pretty(self).map_err(std::io::Error::other)?)
    }
}

/// Games against one opponent rated `opponent` (log-odds), including the virtual draw.
#[derive(Clone, Copy)]
struct Games {
    opponent: f64,
    points: f64,
    count: f64,
}

impl Games {
    fn new(opponent: f64, [w, d, l]: [usize; 3]) -> Self {
        Self { opponent, points: w as f64 + 0.5 * d as f64 + 0.5, count: (w + d + l) as f64 + 1.0 }
    }
}

/// One Newton step for a player rated `x` (log-odds) towards the maximum-likelihood rating, and
/// the Fisher information at `x`.
fn newton_step(x: f64, games: impl Iterator<Item = Games>) -> (f64, f64) {
    let (mut gradient, mut fisher) = (0.0, 0.0);
    for g in games {
        let p = 1.0 / (1.0 + (g.opponent - x).exp());
        gradient += g.points - g.count * p;
        fisher += g.count * p * (1.0 - p);
    }
    let step = if fisher > 0.0 { (gradient / fisher).clamp(-2.0, 2.0) } else { 0.0 };
    (step, fisher)
}

fn stderr(fisher: f64) -> f64 {
    if fisher > 0.0 { ELO_PER_NAT / fisher.sqrt() } else { f64::INFINITY }
}

fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected(diff: f64) -> f64 {
        1.0 / (1.0 + 10f64.powf(-diff / 400.0))
    }

    /// `n` decisive games scoring as expected for a rating difference of `diff` (no draws).
    fn games(diff: f64, n: usize) -> [usize; 3] {
        let wins = (expected(diff) * n as f64).round() as usize;
        [wins, 0, n - wins]
    }

    #[test]
    fn calibration_recovers_ratings_and_estimate_places_a_newcomer() {
        let truth = [0.0, 150.0, 300.0, 600.0];
        let opponents: Vec<OpponentSpec> = (1..=4).map(|depth| OpponentSpec::Minimax { depth }).collect();
        let mut pairings = Vec::new();
        for a in 0..4 {
            for b in a + 1..4 {
                pairings.push(Pairing { a, b, wdl: games(truth[a] - truth[b], 4000) });
            }
        }
        let pool = EloPool::calibrate(opponents.clone(), pairings, 4000);
        for (anchor, t) in pool.anchors.iter().zip(truth) {
            assert!((anchor.rating.elo - t).abs() < 10.0, "{} vs {t}", anchor.rating.elo);
        }
        assert_eq!(pool.anchors[0].rating.stderr, 0.0);

        // A player 400 Elo above the first anchor, seen through its results against two anchors.
        let results = [(&opponents[1], games(400.0 - 150.0, 2000)), (&opponents[3], games(400.0 - 600.0, 2000))];
        let r = pool.estimate(results).unwrap();
        assert!((r.elo - 400.0).abs() < 15.0, "{r:?}");
        assert!(r.stderr > 0.0 && r.stderr < 15.0);
    }

    #[test]
    fn estimate_counts_draws_as_half_and_stays_finite() {
        let pool = EloPool::calibrate(vec![OpponentSpec::Random], Vec::new(), 0);
        // 76 % (here: half the games drawn) is about +200 Elo.
        let r = pool.estimate([(&OpponentSpec::Random, [520, 480, 0])]).unwrap();
        assert!((r.elo - 200.0).abs() < 5.0, "{r:?}");
        // Winning everything gives a large but finite rating thanks to the virtual draw.
        let r = pool.estimate([(&OpponentSpec::Random, [48, 0, 0])]).unwrap();
        assert!(r.elo.is_finite() && r.elo > 500.0, "{r:?}");
        // Opponents outside the pool are ignored.
        assert!(pool.estimate([(&OpponentSpec::Minimax { depth: 2 }, [10, 0, 0])]).is_none());
    }
}
