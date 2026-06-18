import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys


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
        
        self.private_cells = tuple(list(range(0, 5)) + [13, 14, 15])
        self.public_cells = tuple(range(5, 13))

        self.rosettes = (4, 8, 14)
        self.all_cells = tuple(range(0, 16))

        # 2*N is the total number of pieces for both players
        # Each piece position has len(self.all_cells) possible values (0..15),
        # and the dice roll has 5 possible values (0..4).
        self.observation_space = spaces.MultiDiscrete([len(self.all_cells)] * (2 * self.N) + [5])

        # Actions are start positions directly: action 0 moves a piece from
        # position 0, action 1 from position 1, etc. Action 15 is pass.
        self.pass_action = self.scored_cell
        self.action_space = spaces.Discrete(self.scored_cell + 1)

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
        self.last_landed_position = None
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
        pieces = self.player1_loc if self.current_player == 1 else self.player2_loc
        return all(piece == self.scored_cell for piece in pieces)

    def get_legal_moves(self):
        """
        Gets legal actions for current player
        """
        pieces = self.player1_loc if self.current_player == 1 else self.player2_loc
        legal_actions = []

        for start in sorted(set(pieces)):
            if self.roll == 0 or start == self.scored_cell:
                continue

            destination = min(start + self.roll, self.scored_cell)
            # The only pieces that can shere the same cells are the one in the home or score cell, otherwise the action is illegal
            if destination not in (self.home_cell, self.scored_cell) and destination in pieces:
                continue

            legal_actions.append(start)

        return legal_actions or [self.pass_action]

    def update_board(self, action):
        """
        Update current player pieces positions.
        Checks if current player captured any enemy pieces and in case resets their position
        """
        self.last_landed_position = None
        if action == self.pass_action:
            return

        own_pieces = self.player1_loc if self.current_player == 1 else self.player2_loc
        enemy_pieces = self.player2_loc if self.current_player == 1 else self.player1_loc

        start = int(action)
        destination = min(start + self.roll, self.scored_cell)
        own_pieces[own_pieces.index(start)] = destination
        self.last_landed_position = destination

        if destination in self.public_cells and destination in enemy_pieces:
            enemy_pieces[enemy_pieces.index(destination)] = self.home_cell

    def is_on_rosette(self):
        """
        Checks if current player is on a rosette, if not updates the current player
        """
        return self.last_landed_position in self.rosettes
                      
    def move_p2(self):
        self.current_player = 2

        while self.current_player == 2:
            self.roll = self.roll_dice()
            legal_actions = self.get_legal_moves()
            p2_action = int(self.np_random.choice(legal_actions))
            self.update_board(p2_action)

            if self.check_win():
                return True

            if not self.last_landed_position in self.rosettes:
                self.current_player = 1

        return False

    def step(self, action):
        self.current_player = 1
        legal_actions = self.get_legal_moves()

        if action not in legal_actions:
            raise ValueError(f"Illegal action {action} in state {self.state.tolist()}")

        self.update_board(action)
        terminated = self.check_win()
        reward = 1 if terminated else 0

        if not terminated:
            if self.is_on_rosette():
                self.roll = self.roll_dice()
            else:
                terminated = self.move_p2()
                if not terminated:
                    self.roll = self.roll_dice()

        self.current_player = 1
        self.state = np.array(
            self.player1_loc + self.player2_loc + [self.roll],
            dtype=np.int64,
        )
        info = {"legal_actions": [] if terminated else self.get_legal_moves()}
        return self.state, reward, terminated, False, info



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

