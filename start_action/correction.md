# Part2: Implement the game environment

We model the environment using the gymnasium library. First of all in the `__init__` method we define problem specific variables such as the number of pieces per player, the home cell, the scored cell, and tuples with the board numbers of private cells, public cells, and rosettes. Then we define the observation and the action space, which in case of 2 pieces per player are:

- observation space: `MultiDiscrete([16, 16, 16, 16, 5])`
- action space: `Discrete(16)`

The observation has the form `[p11, p12, p21, p22, dice]`, where the first two values are the positions of player 1 pieces, the next two are the positions of player 2 pieces, and the last value is the dice roll. Piece positions go from `0` to `15`: `0` is the home cell, `1` to `14` are board positions, and `15` is the scored cell. The dice roll can be `0, 1, 2, 3, 4`. Actions `0` to `14` mean moving a piece from that position, while action `15` is the pass action.

Also in the `__init__` the `reset` method is called. This method resets simulation variables, so for example it sets players pieces positions to be all in the home cell (so 0), sets the current player playing to 1, rolls the dice, and creates the first state.

In the `step` function we first set the current player to 1. We then check if the agent action is legal and update the board based on the action, so we compute the destination of the piece and, in case it overlaps with one of the enemies on a public cell, we reset the enemy position to the home cell. If the first player wins, the episode terminates with reward 1. If the first player does not win and his piece lands on a rosette, a new dice is rolled and player 1 keeps the turn. If the first player does not land on a rosette, then the action of the second player is picked randomly from the possible legal actions. If the second player lands on a rosette, he keeps playing until he either wins or makes a move that is not on a rosette. If the second player wins, the episode terminates with reward 0; otherwise a new dice is rolled for player 1.
