// mcts/examples/scratch.rs
use mcts::game::*;

fn f1() {
   let mut i : u64 = 1;
   let mask = i;
   i = 2;
   println!("{mask}");
}

fn main() {
    let player = PlayerIndex::<2>::P0;
    println!("Player index: {}", player);
    f1();
}