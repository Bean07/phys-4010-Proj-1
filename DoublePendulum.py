from collections import namedtuple
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

Parameters = namedtuple("Parameters", ["l1", "m1", "l2", "m2", "g"])
DepVariables = namedtuple("DepVariables", ["th1", "th1d", "th2", "th2d"])


def eom_double_pendulum(t, y, *parameters):
    th1, th1d, th2, th2d = y
    l1, m1, l2, m2, g = parameters

    dth = th1 - th2
    den = m1 + m2 * np.sin(dth) ** 2

    th1dd = (-(m1 + m2) * g * np.sin(th1)
             - m2 * l2 * th2d**2 * np.sin(dth)
             + m2 * np.cos(dth) * (g * np.sin(th2)
                                    - l1 * th1d**2 * np.sin(dth))) / (l1 * den)

    th2dd = ((m1 + m2) * (-g * np.sin(th2)
                          + l1 * th1d**2 * np.sin(dth))
             + np.cos(dth) * ((m1 + m2) * g * np.sin(th1)
                              + m2 * l2 * th2d**2 * np.sin(dth))) / (l2 * den)

    return [th1d, th1dd, th2d, th2dd]


def compute_energy(y, parameters):
    th1, th1d, th2, th2d = y
    l1, m1, l2, m2, g = parameters

    T = 0.5 * (m1 + m2) * (l1 * th1d) ** 2 \
        + 0.5 * m2 * (l2 * th2d) ** 2 \
        + m2 * l1 * l2 * th1d * th2d * np.cos(th1 - th2)
    U = -(m1 + m2) * g * l1 * np.cos(th1) \
        - m2 * g * l2 * np.cos(th2)

    return T + U


def make_plots(sol, parameters, t_start=None, t_end=None):
    start_idx = 0 if t_start is None else np.searchsorted(sol.t, t_start, side='left')
    end_idx = len(sol.t) if t_end is None else np.searchsorted(sol.t, t_end, side='right')
    sel = slice(start_idx, end_idx)

    t = sol.t[sel]
    th1, th1d, th2, th2d = sol.y[:, sel]
    x1 = parameters.l1 * np.sin(th1)
    y1 = -parameters.l1 * np.cos(th1)
    x2 = x1 + parameters.l2 * np.sin(th2)
    y2 = y1 - parameters.l2 * np.cos(th2)
    energy = compute_energy(sol.y[:, sel], parameters)
    energy0 = compute_energy(sol.y[:, 0], parameters)
    energy_scale = ((parameters.m1 + parameters.m2) * parameters.g * parameters.l1
                    + parameters.m2 * parameters.g * parameters.l2)

    # Plot trajectories of the two masses
    plt.figure(dpi=120)
    plt.plot(x1, y1, label="Mass 1")
    plt.plot(x2, y2, label="Mass 2")
    plt.plot([0], [0], "ko")  # origin
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Trajectories of the Masses in the Double Pendulum System")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot normalized energy error over time
    plt.figure(dpi=120)
    plt.plot(t, (energy - energy0) / energy_scale, label="Normalized energy error", c="C4", lw=2)
    plt.xlabel("Time (s)")
    plt.ylabel("(Energy - Initial Energy) / Energy Scale")
    plt.title("Normalized Energy Error of the Double Pendulum System")
    plt.grid()
    plt.show()

    # Plot state-space projections for the two arms
    plt.figure(dpi=120)
    plt.plot(th1, th1d, label="Arm 1")
    plt.xlabel("Theta 1 (rad)")
    plt.ylabel("Theta 1 dot (rad/s)")
    plt.title("State-Space Projection of Arm 1 in the Double Pendulum System")
    plt.grid()
    plt.show()

    plt.figure(dpi=120)
    plt.plot(th2, th2d, label="Arm 2")
    plt.xlabel("Theta 2 (rad)")
    plt.ylabel("Theta 2 dot (rad/s)")
    plt.title("State-Space Projection of Arm 2 in the Double Pendulum System")
    plt.grid()
    plt.show()
    
    
    
parameters = Parameters(
    l1 = 0.5,
    m1 = 1.0,
    l2 = 0.5,
    m2 = 1.0,
    g = 9.8,
)

init_conditions = DepVariables(
    th1 = 0,
    th1d = 0,
    th2 = 0.3 * np.pi,
    th2d = 0,
)

t_span = (0.0, 5.0)
# These are output sampling times; solve_ivp chooses adaptive internal timesteps.
t_eval = np.linspace(*t_span, 1001)

sol = solve_ivp(
    eom_double_pendulum, t_span, init_conditions, args=parameters,
    rtol=1e-7, atol=1e-14, t_eval=t_eval, method="DOP853"
)
print(sol.message)
make_plots(sol, parameters, t_start=None, t_end=None)
