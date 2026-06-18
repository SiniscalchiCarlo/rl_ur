# Part2: Implement the game enviroment

We model the enviroment using the gymnasium library. First of all in the __init__ method we define problem specific variables such has tuples with the grud numbers of private and public cells and rosettes. Then we define the observation and the action space, which are:
''fill with numeric values in case of 2 pieces''
Also in the __init__ the reset method is called. This method resets simulation variables, so for example sets players pieces position to be all in the home cell (so 0), sets current player playing to 1.
In the step function we first set the current player to 1, we then update the board based on the agent action, so computes the destination of the piece and in case overlaps with one of the enemies it resets their position. If first player did not won and his piece is not on a rosette, the action, or actions if he steps on a rosette, of the second player is picked from the possible leagal actions.
