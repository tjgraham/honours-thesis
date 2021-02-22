"""Plots the value of a two-player parameterised incompetent game."""

import matrix_solver

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

solver = None
# solver = pulp.GUROBI_CMD()


# --------------------------------------------------------------------------- #
# Game Definition                                                             #
# ----------------------------------------------------------------------------#

m1 = 2  # Player 1 Action Count
m2 = 2  # Player 2 Action Count
F = [
     [3, -2],
     [-2, 1]    
]  # Fully Competent Utility Matrix
def Q1(t): # Player 1 Learning Trajectory
    return [
        [1 / 3 * (1 - t) + t, 2 / 3 * (1 - t)],
        [2 / 3 * (1 - t), 1 / 3 * (1 - t) + t]
    ]
def Q2(t): # Player 2 Learning Trajectory
    return [
        [1 / 4 * (1 - t) + t, 3 / 4 * (1 - t)],
        [3 / 4 * (1 - t), 1 / 4 * (1 - t) + t]
    ]
RESOLUTION = 11


# --------------------------------------------------------------------------- #
# Set-Up                                                                      #
# ----------------------------------------------------------------------------#

# Value of Fully Competent Game
V = matrix_solver.solve(m1, m2, F)[matrix_solver.VALUE]

def R(t, s):  # Utility Matrix under Incompetence
    return [
        [sum(Q1(t)[i][k] * F[k][l] * Q2(s)[j][l] for k in range(m1)
             for l in range(m2)) for j in range(m2)] for i in range(m1)
    ]

X = np.linspace(0, 1, RESOLUTION)  # Player 2 Incompetence Parameters
Y = np.linspace(0, 1, RESOLUTION)  # Player 1 Incompetence Parameters

Z = np.array([
    [round(matrix_solver.solve(
        m1, m2, R(x, y), solver=solver
    )[matrix_solver.VALUE], 2) for y in Y] for x in X
])  # Game Value

X, Y = np.meshgrid(X, Y)


# --------------------------------------------------------------------------- #
# Plot                                                                        #
# ----------------------------------------------------------------------------#

fig = plt.figure()
ax = plt.axes(projection='3d')


bound = max(abs(np.min(Z)), abs(np.max(Z)))

norm = colors.DivergingNorm(vmin=-(bound + 0.1), vcenter=0, vmax=(bound + 0.1))

color_dict = {
    'red': (
        (0.0,  132 / 256, 132 / 256),
        (0.5,  1.0, 1.0),
        (1.0,  53 / 256, 53 / 256)),
    'green': (
        (0.0,  31 / 256, 31 / 256),
        (0.5, 1.0, 1.0),
        (1.0, 78 / 256, 78 / 256)),
    'blue': (
        (0.0,  39 / 256, 39 / 256),
        (0.5,  1.0, 1.0),
        (1.0,  113 / 256, 113 / 256))
}
color_map = colors.LinearSegmentedColormap('ColorMap', color_dict)

# Plot the surface.
surface = ax.plot_surface(
    X, Y, Z, cmap=color_map, linewidth=0.1, edgecolor='black', norm=norm
)

# Set the viewing angle.
ax.view_init(elev=45, azim=225)

# Axes Labels
ax.set_xlabel('Player 2 Learning Parameter $\mu$')
ax.set_ylabel('Player 1 Learning Parameter $\lambda$')
ax.set_zlabel('Game Value $\mathsf{val}(G_{\lambda, \mu})$')

# plt.savefig('incompetent_game_plot.png')
plt.show()
