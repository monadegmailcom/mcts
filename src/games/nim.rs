//! Two-player Misère Nim parameterized by heap count `const HEAPS: usize`
//! (port of C++ `template <size_t N> nim::State<N>` in `games/nim.h`).

use crate::game::{Game, GameRules, GameStatus, PlayerIndex};

/// Remove `count >= 1` stones from `heap_index`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NimMove {
    pub heap_index: u16,
    pub count: u16,
}

pub const NO_MOVE: NimMove = NimMove {
    heap_index: 0,
    count: 0,
};

/// Zero-sized game rules marker for `HEAPS`-pile Misère Nim (`N = 2` players).
#[derive(Clone, Copy, Debug)]
pub struct Nim<const HEAPS: usize>;

pub type NimState<const HEAPS: usize> = [u16; HEAPS];
pub type NimGame<const HEAPS: usize> = Game<Nim<HEAPS>, 2>;

/// Lazy, zero-allocation move iterator across heaps and removal counts.
pub struct NimMoveIter<'a, const HEAPS: usize> {
    state: &'a NimState<HEAPS>,
    heap: usize,
    next_count: u16,
}

impl<const HEAPS: usize> Iterator for NimMoveIter<'_, HEAPS> {
    type Item = NimMove;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.heap < HEAPS {
            if self.next_count <= self.state[self.heap] {
                let mv = NimMove {
                    heap_index: self.heap as u16,
                    count: self.next_count,
                };
                self.next_count += 1;
                return Some(mv);
            }
            self.heap += 1;
            self.next_count = 1;
        }
        None
    }
}

impl<const HEAPS: usize> GameRules<2> for Nim<HEAPS> {
    type Move = NimMove;
    type State = NimState<HEAPS>;
    type MoveIter<'a> = NimMoveIter<'a, HEAPS>;

    #[inline]
    fn valid_moves<'a>(state: &'a Self::State, _player: PlayerIndex<2>) -> Self::MoveIter<'a> {
        NimMoveIter {
            state,
            heap: 0,
            next_count: 1,
        }
    }

    #[inline]
    fn write_valid_moves(
        state: &Self::State,
        _player: PlayerIndex<2>,
        out: &mut Vec<Self::Move>,
    ) {
        out.clear();
        for (heap, &stones) in state.iter().enumerate() {
            for count in 1..=stones {
                out.push(NimMove {
                    heap_index: heap as u16,
                    count,
                });
            }
        }
    }

    #[inline]
    fn apply(state: &Self::State, _player: PlayerIndex<2>, mv: Self::Move) -> Self::State {
        let heap = mv.heap_index as usize;
        assert!(heap < HEAPS, "heap index {heap} out of range for {HEAPS}");
        assert!(
            mv.count >= 1 && mv.count <= state[heap],
            "invalid Nim move count {} on heap size {}",
            mv.count,
            state[heap]
        );
        let mut next = *state;
        next[heap] -= mv.count;
        next
    }

    #[inline]
    fn status(state: &Self::State, current_player: PlayerIndex<2>) -> GameStatus<2> {
        // Misère convention (matching `games/nim.h`): when all heaps are empty,
        // the previous player took the last object and lost, so `current_player` wins.
        if state.iter().all(|&stones| stones == 0) {
            GameStatus::single_winner(current_player)
        } else {
            GameStatus::Ongoing
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nim_progression_matches_cpp_test() {
        let mut game = NimGame::<2>::new(PlayerIndex::PLAYER1, [1, 2]);
        assert_eq!(
            game.valid_moves().collect::<Vec<_>>(),
            vec![
                NimMove {
                    heap_index: 0,
                    count: 1
                },
                NimMove {
                    heap_index: 1,
                    count: 1
                },
                NimMove {
                    heap_index: 1,
                    count: 2
                },
            ]
        );

        game = game.apply(NimMove {
            heap_index: 1,
            count: 1,
        });
        assert_eq!(game.current_player(), PlayerIndex::PLAYER2);

        game = game.apply(NimMove {
            heap_index: 1,
            count: 1,
        });
        assert_eq!(game.current_player(), PlayerIndex::PLAYER1);

        // Player 1 is forced to take the final stone -> Player 2 wins!
        game = game.apply(NimMove {
            heap_index: 0,
            count: 1,
        });
        assert_eq!(game.current_player(), PlayerIndex::PLAYER2);
        assert_eq!(
            game.status(),
            GameStatus::single_winner(PlayerIndex::PLAYER2)
        );
    }
}
