"""Solves matrix games using a linear program."""

import pulp

# Indices of `solve` outputs.
VALUE = 0  # Game Value
PLAYER1 = 1  # Player 1 Optimal Strategy
PLAYER2 = 2  # Player 2 Optimal Strategy

def solve(m1, m2, R, solver=None):
    """Finds optimal solutions and the game value of a matrix game.
    
    Args:
        m1 (int): The number of actions available to Player 1.
        m2 (int): The number of actions available to Player 2.
        R (List[List[float]]): A utility matrix with m1 rows and m2 columns.
        solver: The solver to be used.
    
    Returns: Tuple[float, Tuple[float], Tuple[float]]
        A tuple contianing the game value, a optimal strategy for Player 1, and
        an optimal strategy for Player 2, respectively.
    """
    # Construct a linear program to find the game value and optimal strategies.
    model = pulp.LpProblem('MatrixGame', pulp.LpMaximize)
    
    # Variables
    v = pulp.LpVariable('Value')  # Game Value
    x = [
        pulp.LpVariable('Row{}'.format(i), 0) for i in range(m1)
    ]  # Player 1 Optimal Strategy
    
    # Objective Function
    model += v
    
    # Constraints
    y = []  # Player 2 Optimal Strategy
    for j in range(m2):
        c = pulp.LpConstraint(
            v - pulp.lpSum(x[i] * R[i][j] for i in range(m1)),
            sense=pulp.LpConstraintLE, rhs=0,  name='Column{}'.format(j)
        )  # The dual values of these constraints give Player 2's strategy.
        y.append(c)
        model += c

    model += pulp.LpConstraint(pulp.lpSum(x[i] for i in range(m1)), rhs=1)

    model.solve(solver=solver)

    return (v.varValue, tuple(x[i].varValue for i in range(m1)),
            tuple(y[j].pi for j in range(m2)))
