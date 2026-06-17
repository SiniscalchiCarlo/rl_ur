import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
import pandas as pd
import matplotlib.pyplot as plt
import sys
import itertools
from pydantic import BaseModel


# Royal game of Ur
class RoyalGameOfUr(gym.Env):

    def __init__(self, N):
        super().__init__()
        if N <= 0:
            raise ValueError("N must be a positive integer")

        # Number of pieces per player.
        self.N = N

        # Board constants used by the transition logic. Positions are encoded as:
        # 0 = not entered, 1..14 = board path, 15 = scored.
        self.home_cell = 0
        self.scored_cell = 15
        self.private_cells = tuple(list(range(0, 5)) + [14, 15])
        self.public_cells = tuple(range(5, 14))
        self.rosettes = (4, 8, 14)
        self.all_cells = tuple(range(0, 16))

        # 2*N is the total number of pieces for both players
        # Each piece position has len(self.all_cells) possible values (0..15),
        # and the dice roll has 5 possible values (0..4).
        self.observation_space = spaces.MultiDiscrete([len(self.all_cells)] * (2 * self.N) + [5])

        # Use action ids for Gymnasium. Id 0 is pass; all other ids decode to
        # the start position of the piece to move. The destination is determined
        # by the current dice roll in the state. Illegal starts are filtered in step.
        self.pass_action = 0
        self.actions = [None] + list(self.all_cells)
        self.action_to_id = {action: action_id for action_id, action in enumerate(self.actions)}
        self.action_space = spaces.Discrete(len(self.actions))

        self.reset()
        
    """
    The initialisation function is already given above, but you should still fill in the other elements of this function.
    The other functions that should be defined are
    
    1. The reset function, which should set the clear the board, set the number of scored pieces per player to zero, and start
    with the turn of player 1
    2. The step function, which gives the next state, reward, and done given the current state, roll, and action.
    (3.) It will be very useful to define a function that gives the possible moves for the current player given the board state
    and roll.
    (4.) It may also be useful to define a function that rolls the dice, but you can also make this part of the step function.
    
    A render function has been given which allows you to visualise episodes of the game being played.
    """
    
    def roll_dice(self):
        return int(self.np_random.binomial(4, 0.5))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player1_loc = [self.home_cell] * self.N
        self.player2_loc = [self.home_cell] * self.N
        self.current_player = 1
        self.roll = self.roll_dice()
        self.state = np.array(
            self.player1_loc + self.player2_loc + [self.roll],
            dtype=np.int64,
        )

        return self.state, {}

    def check_win(self):
        """
        Checks if current player won
        """

    def get_legal_moves(self):
        """
        Gets legal actions for current player
        """

    def update_board(self, action):
        """
        Update current player pieces positions.
        Checks if current player captured any enemy pieces and in case resets their position
        """

    def is_on_rosette(self):
        """
        Checks if current player is on a rosette, if not updates the current player
        """
                      

    def step(self, action):
        self.roll = self.roll_dice()
        legal_actions = self.get_legal_moves(action)

        if self.current_player == 1:
            if self.action and self.action in legal_actions:
                self.update_board(action)

        else:
            p2_action = self.np_random.choice(legal_actions)
            if p2_action:
                self.update_board(action)


        self.check_win()
        self.is_on_rosette()

    def render(self, player1_loc, player2_loc):
        """
        The input for this render functions are lists or sets player1_loc and player2_loc.
        These should contain the location of the pieces for the respective player.
        The location should be denoted by the number of the square on the path of that player.
        More precisely, we can number the squares for both players on their path from 1 to 14, with 1 the first square on the
        path, and 14 the last square on the path, which is the final one before leaving the board.
        If player1_loc then equals [2, 4] for example, then he has two pieces in play. The first is on square 2, i.e., the
        second square on their path, while the second is on square 4, i.e., the fourth square on their path.
        """
        
        # Obtain states
        seq_1 = [(i, 1) for i in range(4, 0, -1)] + [(i, 2) for i in range(1, 9)] + [(8, 1), (7, 1)]
        seq_2 = [(i, 3) for i in range(4, 0, -1)] + [(i, 2) for i in range(1, 9)] + [(8, 3), (7, 3)]
        self.encode_states = [{idx: s for s, idx in zip(seq_1, range(1, 15))},
                          {idx: s for s, idx in zip(seq_2, range(1, 15))}]
        board_states_1 = [self.encode_states[0][i] for i in player1_loc]
        board_states_2 = [self.encode_states[1][i] for i in player2_loc]
        
        
        outfile = sys.stdout
        for i in range(1, 9):
            for j in range(1, 4):
                if (i, j) in board_states_1:
                    output = " X "
                elif (i, j) in board_states_2:
                    output = " O "
                elif (i, j) in [(5, 1), (5, 3)]:
                    output = " S "
                elif (i, j) in [(6, 1), (6, 3)]:
                    output = " E "
                elif (i, j) in [(1, 1), (1, 3), (7, 1), (7, 3), (4, 2)]:
                    output = " R "
                else:
                    output = " - "
    
                if j == 1:
                    output = output.lstrip()
                if j == 3:
                    output = output.rstrip()
                    output += "\n"
    
                outfile.write(output)
        outfile.write("\n")

               
        return

