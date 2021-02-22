"""Solves incremental learning games using backward induction."""

import matrix_solver

import itertools
import math
import time

import pulp
import matplotlib.pyplot as plt

solver = None
# solver = pulp.GUROBI_CMD()


# --------------------------------------------------------------------------- #
# Game Definition                                                             #
# ----------------------------------------------------------------------------#
m1 = 3  # Player 1 Action Count
m2 = 3  # Player 2 Action Count
F = [
    [  0,    40, 100],
    [ -40,    0, 100],
    [-100, -100,   0]
]  # Fully Competent Utility Matrix
def Q1(t): # Player 1 Learning Trajectory
    return [
        [3 / 10 * (1 - t) + t, 1 / 10 * (1 - t), 3 / 5 * (1 - t)],
        [1 / 10 * (1 - t), 3 / 5 * (1 - t) + t, 3 / 10 * (1 - t)],
        [0, 0, 1]
    ]
def Q2(t): # Player 2 Learning Trajectory
    return [
        [3 / 10 * (1 - t) + t, 1 / 10 * (1 - t), 3 / 5 * (1 - t)],
        [1 / 10 * (1 - t), 3 / 5 * (1 - t) + t, 3 / 10 * (1 - t)],
        [0, 0, 1]
    ]

n1 = 6  # Player 1 Learning Parameter Count
n2 = 6  # Player 2 Learning Parameter Count
Lambda = [1 / (n1 - 1) * i for i in range(n1)]  # Player 1 Learning Parameters
Mu = [1 / (n2 - 1) * j for j in range(n2)]  # Player 2 Learning Parameters

cost1 = {
    (i, j): 10 for i in range(n1) for j in range(n2) 
}  # Player 1 Learning Costs
cost2 = {
     (i, j): 10 for i in range(n1) for j in range(n2)
}  # Player 2 Learning Costs

discount = 19 / 20 # Discount Factor


# --------------------------------------------------------------------------- #
# Set-Up                                                                      #
# ----------------------------------------------------------------------------#
states = list(itertools.product(range(n1), range(n2)))  # State Space
states.sort()  # Arrange states in lexicographical order.

def R(t, s):  # Utility Matrix under Incompetence
    return [
        [sum(Q1(t)[i][k] * F[k][l] * Q2(s)[j][l] for k in range(m1)
             for l in range(m2)) for j in range(m2)] for i in range(m1)
    ]

# Calculate the incompetent game value at every state.
value_time = time.perf_counter()
value = {
    (i, j): matrix_solver.solve(
        m1, m2, R(Lambda[i], Mu[j]), solver=solver
    )[matrix_solver.VALUE] for i in range(n1) for j in range(n2)
}
value_time = time.perf_counter() - value_time


# --------------------------------------------------------------------------- #
# Incremental Learning Game Solver                                            #
# ----------------------------------------------------------------------------#

# Indices used for output.
PLAYER1 = 0
PLAYER2 = 1

STAY = 0
LEARN = 1
VALUE = 2

def solve(state):
    """Solves an incremental learning game starting at the given state.
    
    Args:
        state: The current state of the game.
    
    Returns:
        A list of dictionaries which each assign future states to a tuple
        containing the expected values and optimal strategies for Player 1 and
        Player 2, respectively.
    """
    # Check that a valid state was provided.
    if state not in states:
        raise ValueError('The state {} could not be found.'.format(state))

    index = states.index(state)
    if index == len(states) - 1:
        # This is the last state.
        return [{state: ((1.0, 0.0, value[state] / (1 - discount)),
                         (1.0, 0.0, -(value[state] / (1 - discount))))}]

    # Find every equilibrium of the future states.
    future_eqbs = solve(states[index + 1])

    # Given each future equilibrium, find every extension to the current state.
    return [eqb for future_eqb in future_eqbs
            for eqb in _induction_step(state, future_eqb)]

# --------------------------------------------------------------------------- #
# Induction Step                                                              #
# ----------------------------------------------------------------------------#

def _V(state, future_eqb):
    """Defines the V constants at the current state.
    
    Args:
        state: The current state.
        future_eqb: An equilibrium for the future states.
    
    Returns:
        A dictionary containing the V constants for each player and action.
    """
    i, j = state
    A = {0} if i == n1 - 1 else {0, 1}  # Player 1 Actions
    B = {0} if j == n2 - 1 else {0, 1}  # Player 2 Actions

    V = {}
    for a, b in itertools.product(A, B):
        if a == 0 and b == 0:
            V[PLAYER1, a, b] = value[state]
            V[PLAYER2, a, b] = -value[state]
            continue
        
        V[PLAYER1, a, b] = (value[state] - a * cost1[state] + discount 
                            * future_eqb[i + a, j + b][PLAYER1][VALUE])
        V[PLAYER2, a, b] = (-value[state] - b * cost2[state]  + discount 
                            * future_eqb[i + a, j + b][PLAYER2][VALUE])

    return V


def _induction_step(state, future_eqb):
    """Finds all the equilibria at the current state given future behaviour.
    
    Delegates to `_player1_incompetent`, `_player2_incompetent`, or
    `_both_incompetent` depending on the current state.
    
    Args:
        state: The current state.
        future_eqb: An equilibrium for the future states.
    
    Returns:
        A list of equilibria that extend the future equilibrium to the current
        state.
    """
    i, j = state

    if i != n1 - 1 and j == n2 - 1:  # Only Player 1 is incompetent.
        case = _player1_incompetent

    elif i == n1 - 1 and j != n2 - 1: # Only Player 2 is incompetent.
        case = _player2_incompetent

    else:  # Both Player 1 and Player 2 are incompetent.
        case = _both_incompetent

    # Find the equilibrium at the current state given the future behaviour.
    eqbs = case(state, future_eqb)

    # Combine the equilibria at the current state with the future equilibrium.
    for eqb in eqbs:
        eqb.update(future_eqb)

    return eqbs


def _player1_incompetent(state, future_eqb):
    """Finds all the equilibria at the current state given future behaviour.

    Helper function for handling the case where `i != n1` and `j == n2`.

    Args:
        state: The current state.
        future_eqb: An equilibrium for the future states.
    
    Returns:
        A list of equilibria that extend the future equilibrium to the current
        state.
    """
    V = _V(state, future_eqb)  # A dictionary containing the V constants.

    discounted_value1 = (lambda p: 
        (p * V[PLAYER1, STAY, STAY] + (1 - p) * V[PLAYER1, LEARN, STAY])
        / (1 - discount * p)
    )  # Player 1's discounted value of a current strategy.
    discounted_value2 = (lambda p:
        (p * V[PLAYER2, STAY, STAY] + (1 - p) * V[PLAYER2, LEARN, STAY])
        / (1 - discount * p)
    )  # Player 2's discounted value of a current strategy.
    
    eqbs = []  # The set of current equilibria.
    
    # Compare the discounted values of the pure strategies.
    for p in [1.0, 0.0]:
        if (discounted_value1(p) >= discounted_value1(1 - p)):
            # The probability p for not learning produces an equilibrium.
            eqbs.append({state: ((p, 1 - p, discounted_value1(p)),
                                 (1.0, 0.0, discounted_value2(p)))})

    return eqbs


def _player2_incompetent(state, future_eqb):
    """Finds all the equilibria at the current state given future behaviour.
    
    Helper function for handling the case where `i == n1` and `j != n2`.
    
    Args:
        state: The current state.
        future_eqb: An equilibrium for the future states.
    
    Returns:
        A list of equilibria that extend the future equilibrium to the current
        state.
    """
    V = _V(state, future_eqb)  # A dictionary containing the V constants.

    discounted_value1 = (lambda q: 
        (q * V[PLAYER1, STAY, STAY] + (1 - q) * V[PLAYER1, STAY, LEARN])
        / (1 - discount * q)
    )  # Player 1's discounted value of a current strategy.
    discounted_value2 = (lambda q:
        (q * V[PLAYER2, STAY, STAY] + (1 - q) * V[PLAYER2, STAY, LEARN])
        / (1 - discount * q)
    )  # Player 2's discounted value of a current strategy.
    
    eqbs = []  # The set of current equilibria.
    
    # Compare the discounted values of the pure strategies.
    for q in [1.0, 0.0]:
        if (discounted_value2(q) >= discounted_value2(1 - q)):
            # The probability p for not learning produces an equilibrium.
            eqbs.append({state: ((1.0, 0.0, discounted_value1(q)),
                                 (q, 1 - q, discounted_value2(q)))})

    return eqbs


def roots(a, b, c):
    """Finds the roots of f(x) = ax^2 + bx + c inside the interval (0, 1)."""
    if a == 0: 
        if b == 0: # f(x) = c
            roots = []
    
        else: # f(x) = bx + c
            roots = [-(c / b)]
    
    elif b ** 2 - 4 * a * c >= 0: # f(x) = ax^2 + bx + c
        roots = [(-b + math.sqrt(b ** 2 - 4 * a * c)) / (2 * a),
                 (-b - math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)]

    else:
        roots = []

    return [x for x in roots if x > 0 and x < 1]


def _both_incompetent(state, future_eqb):
    """Finds all the equilibria at the current state given future behaviour.
    
    Helper function for handling the case where `i != n1` and `j != n2`.
    
    Args:
        state: The current state.
        future_eqb: An equilibrium for the future states.
    
    Returns:
        A list of equilibria that extend the future equilibrium to the current
        state.
    """
    V = _V(state, future_eqb)  # A dictionary containing the V constants.

    discounted_value1 = (lambda p, q: 
        (p * q * V[PLAYER1, STAY, STAY] 
         + p * (1 - q) * V[PLAYER1, STAY, LEARN]
         + (1 - p) * q * V[PLAYER1, LEARN, STAY]
         + (1 - p) * (1 - q) * V[PLAYER1, LEARN, LEARN]) 
        / (1 - discount * p * q)
    )  # Player 1's discounted value of a current strategy.
    discounted_value2 = (lambda p, q: 
        (p * q * V[PLAYER2, STAY, STAY] 
         + p * (1 - q) * V[PLAYER2, STAY, LEARN]
         + (1 - p) * q * V[PLAYER2, LEARN, STAY]
         + (1 - p) * (1 - q) * V[PLAYER2, LEARN, LEARN]) 
        / (1 - discount * p * q)
    )  # Player 2's discounted value of a current strategy.

    eqbs = []  # The set of current equilibria.

    # Find any pure strategy equilibria.   
    for p, q in itertools.product([1.0, 0.0], [1.0, 0.0]):
        if (discounted_value1(p, q) >= discounted_value1(1 - p, q)
                and discounted_value2(p, q) >= discounted_value2(p, 1 - q)):
            # The probabilities (p, q) for not learning produce an equilibrium.
            eqbs.append({state: ((p, 1 - p, discounted_value1(p, q)),
                                 (q, 1 - q, discounted_value2(p, q)))})
            
    # Find any mixed strategy equilibria.
    roots1 = roots(
        discount * (V[PLAYER2, STAY, LEARN] - V[PLAYER2, LEARN, LEARN]),
        (V[PLAYER2, STAY, STAY] - V[PLAYER2, STAY, LEARN] -
         V[PLAYER2, LEARN, STAY] + (1 + discount) * V[PLAYER2, LEARN, LEARN]),
        V[PLAYER2, LEARN, STAY] - V[PLAYER2, LEARN, LEARN]
    )  # Mixed strategy equilibrium probabilities for Player 1.
    roots2 = roots(
        discount * (V[PLAYER1, LEARN, STAY] - V[PLAYER1, LEARN, LEARN]),
        (V[PLAYER1, STAY, STAY]  - V[PLAYER1, STAY, LEARN] -
         V[PLAYER1, LEARN, STAY] + (1 + discount) * V[PLAYER1, LEARN, LEARN]),
        V[PLAYER1, STAY, LEARN] - V[PLAYER1, LEARN, LEARN]
    )  # Mixed strategy equilibrium probabilites for Player 2.

    # Combine every pair of roots to produce the mixed strategies.
    for p, q in itertools.product(roots1, roots2):
        eqbs.append({state: ((p, 1 - p, discounted_value1(p, q)), 
                             (q, 1 - q, discounted_value2(p, q)))})
        
    return eqbs


# --------------------------------------------------------------------------- #
# Show                                                                        #
# ----------------------------------------------------------------------------#
    
TOL = 1e-6

def show(eqb):
    """Displays an equilibrium solution to an incremental learning game.
    
    Args:
        eqb: An equilibrium solution to an incremental learning game.
    """
    fig = plt.figure()
    ax = plt.axes()

    dx = 0.3 / (n1 - 1)
    dy = 0.3 / (n1 - 1)

    for state in eqb:
        i, j = state
        x = i / (n1 - 1)
        y = j / (n2 - 1)
        xx = (i + 1) / (n1 - 1)
        yy = (j + 1) / (n2 - 1) 
        
        ax.plot(x, y,'k.') 
        
        if not (math.isclose(eqb[state][PLAYER1][LEARN], 0, abs_tol=TOL)
                or math.isclose(eqb[state][PLAYER2][STAY], 0, abs_tol=TOL)):
                ax.arrow(x + dx, y, xx - x - 2 * dx, 0, 
                         head_width=0.02, fc='k', ec='k')

        if not (math.isclose(eqb[state][PLAYER1][STAY], 0, abs_tol=TOL)
                or math.isclose(eqb[state][PLAYER2][LEARN], 0, abs_tol=TOL)):
                ax.arrow(x, y + dy, 0, yy - y - 2 * dy,  
                         head_width=0.02, fc='k', ec='k')

        if not (math.isclose(eqb[state][PLAYER1][LEARN], 0, abs_tol=TOL)
                or math.isclose(eqb[state][PLAYER2][LEARN], 0, abs_tol=TOL)):
                ax.arrow(x + dx, y + dy, xx - x - 2 * dx, yy - y - 2 * dy,  
                         head_width=0.02, fc='k', ec='k')

    ax.set_xlabel('Player 1 Learning')
    ax.set_ylabel('Player 2 Learning')
    
    plt.savefig('incremental_game_plot.png', dpi=300)

# --------------------------------------------------------------------------- #
# Main                                                                        #
# ----------------------------------------------------------------------------#

# Solve the incremental learning game from the initial state.
solve_time = time.perf_counter()
eqbs = solve((0, 0))
solve_time = time.perf_counter() - solve_time

print('Value Calculation Time: {}s'.format(round(value_time, 2)))
print('Backward Induction Time: {}s'.format(round(solve_time, 2)))
print('Found {} Equilibrium Solution(s)'.format(len(eqbs)))
print()
print('Run `show(eqbs[i])` to view the i\'th equilibrium.')
