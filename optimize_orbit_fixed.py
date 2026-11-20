import numpy as np
import matplotlib.pyplot as plt
import csv
import os

from scipy.optimize import minimize
from tqdm.auto import tqdm


# ============================================================
# 1. SYSTEM PARAMETERS
# ============================================================

G = 6.67430e-11
AU = 1.495978707e11
YEAR = 365.25 * 24 * 3600
M_SUN = 1.98847e30

# 1 AU/year^2 expressed in m/s^2
AU_YEAR2_TO_MS2 = AU / YEAR**2


class System:

    def __init__(
        self,
        star_mass=1.0,       # Solar masses (gravity acting on the ROCKET only)
        rA=1.0,              # AU
        rB=3.0,             # AU
        phaseA=0.0,          # radians
        phaseB=4.5,          # radians
        thrust=0.0098           # m/s^2
    ):

        # Gravity acting on the rocket
        self.mu = 4 * np.pi**2 * star_mass

        self.rA = rA
        self.rB = rB

        self.phaseA = phaseA
        self.phaseB = phaseB

        # Maximum thrust acceleration, AU/year^2
        self.a = thrust / AU_YEAR2_TO_MS2

        # Planet orbital angular velocities
        # Here planets orbit as if star_mass = 1 solar mass
        self.wA = np.sqrt(4 * np.pi**2 / self.rA**3)
        self.wB = np.sqrt(4 * np.pi**2 / self.rB**3)

        self.periodA = 2 * np.pi / self.wA
        self.periodB = 2 * np.pi / self.wB

    def planet(self, t, which):

        if which == "A":
            r, w, phase = self.rA, self.wA, self.phaseA
        else:
            r, w, phase = self.rB, self.wB, self.phaseB

        angle = phase + w * t

        position = np.array([
            r * np.cos(angle),
            r * np.sin(angle)
        ])

        velocity = np.array([
            -r * w * np.sin(angle),
             r * w * np.cos(angle)
        ])

        return position, velocity

    def print_parameters(self):

        print("\n--- System Parameters ---")
        print(f"Star mass:       {self.mu / (4*np.pi**2):.6f} solar masses")
        print(f"Planet A radius: {self.rA:.6f} AU")
        print(f"Planet B radius: {self.rB:.6f} AU")
        print(f"Planet A omega:  {self.wA:.6f} rad/year")
        print(f"Planet B omega:  {self.wB:.6f} rad/year")
        print(f"Planet A period: {self.periodA:.6f} years")
        print(f"Planet B period: {self.periodB:.6f} years")
        print(f"Maximum thrust:  {self.a:.3f} AU/year^2")
        print("-------------------------\n")


# ============================================================
# 2. ROCKET EQUATIONS OF MOTION
# ============================================================

def dynamics(state, theta, throttle, sys):

    x, y, vx, vy = state

    radius = np.sqrt(x*x + y*y)

    # Star gravity
    gravity = -sys.mu / radius**3

    ax_gravity = gravity * x
    ay_gravity = gravity * y

    # Thrust: throttle ranges from 0 to 1
    ax_thrust = throttle * sys.a * np.cos(theta)
    ay_thrust = throttle * sys.a * np.sin(theta)

    ax = ax_gravity + ax_thrust
    ay = ay_gravity + ay_thrust

    return np.array([
        vx,
        vy,
        ax,
        ay
    ])


# ============================================================
# 3. CSV EXPORT
# ============================================================

def export_solution_csv(
    sys, X, theta, throttle, T,
    filename="orbit_data.csv"
):

    N = len(theta)
    times = np.linspace(0, T, N + 1)

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "node",
            "time_years",

            "rocket_x_AU",
            "rocket_y_AU",
            "rocket_vx_AU_per_year",
            "rocket_vy_AU_per_year",

            "planet_A_x_AU",
            "planet_A_y_AU",
            "planet_A_vx_AU_per_year",
            "planet_A_vy_AU_per_year",

            "planet_B_x_AU",
            "planet_B_y_AU",
            "planet_B_vx_AU_per_year",
            "planet_B_vy_AU_per_year",

            "thrust_angle_rad",
            "throttle",

            "thrust_ax_AU_per_year2",
            "thrust_ay_AU_per_year2",

            "gravity_ax_AU_per_year2",
            "gravity_ay_AU_per_year2",

            "total_ax_AU_per_year2",
            "total_ay_AU_per_year2"
        ])

        for i in range(N + 1):

            t = times[i]

            pA, vA = sys.planet(t, "A")
            pB, vB = sys.planet(t, "B")

            if i < N:

                angle = theta[i]
                u = throttle[i]

                thrust_ax = u * sys.a * np.cos(angle)
                thrust_ay = u * sys.a * np.sin(angle)

                total_acc = dynamics(
                    X[i], angle, u, sys
                )[2:]

                total_ax, total_ay = total_acc

                radius = np.linalg.norm(X[i, :2])

                gravity_acc = (
                    -sys.mu * X[i, :2] / radius**3
                )

                gravity_ax, gravity_ay = gravity_acc

            else:

                angle = ""
                u = ""
                thrust_ax = ""
                thrust_ay = ""
                gravity_ax = ""
                gravity_ay = ""
                total_ax = ""
                total_ay = ""

            writer.writerow([
                i,
                t,

                *X[i],

                *pA,
                *vA,

                *pB,
                *vB,

                angle,
                u,

                thrust_ax,
                thrust_ay,

                gravity_ax,
                gravity_ay,

                total_ax,
                total_ay
            ])

    print(f"CSV exported to: {os.path.abspath(filename)}")


# ============================================================
# 4. OPTIMAL CONTROL PROBLEM
# ============================================================

def _grav_jac(pos, mu):
    """d(gravity acceleration)/d(position), 2x2 (only used if mu != 0)."""
    x, y = pos
    r = np.sqrt(x * x + y * y)
    return mu * np.array([
        [3 * x * x / r**5 - 1 / r**3, 3 * x * y / r**5],
        [3 * x * y / r**5,            3 * y * y / r**5 - 1 / r**3],
    ])


def _initial_guess(sys, N, tau):
    """Physically sensible starting point (all in scaled units, max thrust = 1):
    burn toward where planet B will be, flip at the midpoint, brake.
    The rocket states are *integrated* from those controls, so the
    collocation defects start out ~satisfied instead of being garbage."""
    wB, mu = sys.wB * tau, sys.mu * tau**2

    def planetB_pos(t):
        ang = sys.phaseB + wB * t
        return sys.rB * np.array([np.cos(ang), np.sin(ang)])

    pA0 = sys.rA * np.array([np.cos(sys.phaseA), np.sin(sys.phaseA)])
    vA0 = sys.rA * sys.wA * tau * np.array([-np.sin(sys.phaseA),
                                            np.cos(sys.phaseA)])

    # flip-and-burn: d = a T^2 / 4  ->  T = 2 sqrt(d / a), with a = 1 here.
    # B moves, so iterate T <-> d(T).
    T = 2.0
    for _ in range(30):
        T = 2 * np.sqrt(np.linalg.norm(planetB_pos(T) - pA0))
    direction = planetB_pos(T) - pA0
    direction /= np.linalg.norm(direction)

    U = np.zeros((N, 2))
    U[: N // 2] = direction
    U[N // 2:] = -direction

    def f(s, u):
        r = np.linalg.norm(s[:2])
        return np.concatenate([s[2:], -mu * s[:2] / r**3 + u])

    h = T / N
    X = np.zeros((N + 1, 4))
    X[0] = np.concatenate([pA0, vA0])
    for i in range(N):                                   # RK4
        k1 = f(X[i], U[i])
        k2 = f(X[i] + h / 2 * k1, U[i])
        k3 = f(X[i] + h / 2 * k2, U[i])
        k4 = f(X[i] + h * k3, U[i])
        X[i + 1] = X[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return X, U, T


def optimize_orbit(
    sys,
    N=2500,
    T_guess=None,          # years; default = flip-and-burn estimate
    max_iterations=6000,
    seed=0                 # unused now (no random initial guess); kept for compatibility
):

    # --------------------------------------------------------
    # Nondimensionalise: length = 1 AU, time unit chosen so that
    # max thrust = 1.  (Original: thrust ~ 65,000 AU/yr^2, T ~ 0.008 yr,
    # velocities ~ 250 AU/yr  ->  badly scaled for the optimizer.)
    # --------------------------------------------------------
    tau = 1.0 / np.sqrt(sys.a)            # years per scaled time unit
    mu = sys.mu * tau**2
    wB = sys.wB * tau

    pA0 = sys.rA * np.array([np.cos(sys.phaseA), np.sin(sys.phaseA)])
    vA0 = sys.rA * sys.wA * tau * np.array([-np.sin(sys.phaseA),
                                            np.cos(sys.phaseA)])

    def planetB(t):
        ang = sys.phaseB + wB * t
        p = sys.rB * np.array([np.cos(ang), np.sin(ang)])
        v = sys.rB * wB * np.array([-np.sin(ang), np.cos(ang)])
        return p, v, -wB**2 * p            # position, velocity, acceleration

    X0, U0, T0 = _initial_guess(sys, N, tau)
    if T_guess is not None:
        T0 = T_guess / tau

    # Variable layout: X (4(N+1)), U (2N, thrust VECTOR in units of max thrust), T
    nX, nU = 4 * (N + 1), 2 * N
    z0 = np.concatenate([X0.ravel(), U0.ravel(), [T0]])

    def unpack(z):
        return z[:nX].reshape(N + 1, 4), z[nX:nX + nU].reshape(N, 2), z[-1]

    def vel_and_grav(X):
        pos = X[:, :2]
        if mu == 0:
            return X[:, 2:], np.zeros_like(pos)
        r3 = np.linalg.norm(pos, axis=1)**3
        return X[:, 2:], -mu * pos / r3[:, None]

    # --------------------------------------------------------
    # Equality constraints (+ analytic Jacobian)
    # --------------------------------------------------------
    def constraints(z):
        X, U, T = unpack(z)
        h = T / N
        pB, vB, _ = planetB(T)
        v, g = vel_and_grav(X)
        f0 = np.hstack([v[:-1], g[:-1] + U])
        f1 = np.hstack([v[1:],  g[1:] + U])
        defects = X[1:] - X[:-1] - 0.5 * h * (f0 + f1)
        return np.concatenate([
            X[0] - np.concatenate([pA0, vA0]),
            X[-1] - np.concatenate([pB, vB]),
            defects.ravel(),
        ])

    def constraints_jac(z):
        X, U, T = unpack(z)
        h = T / N
        J = np.zeros((8 + 4 * N, len(z)))
        J[0:4, 0:4] = np.eye(4)
        J[4:8, 4 * N:4 * N + 4] = np.eye(4)
        pB, vB, aB = planetB(T)
        J[4:8, -1] = -np.concatenate([vB, aB])
        v, g = vel_and_grav(X)
        I4 = np.eye(4)
        for i in range(N):
            r0 = 8 + 4 * i
            Fx0 = np.zeros((4, 4)); Fx0[:2, 2:] = np.eye(2)
            Fx1 = np.zeros((4, 4)); Fx1[:2, 2:] = np.eye(2)
            if mu != 0:
                Fx0[2:, :2] = _grav_jac(X[i, :2], mu)
                Fx1[2:, :2] = _grav_jac(X[i + 1, :2], mu)
            J[r0:r0 + 4, 4 * i:4 * i + 4] = -I4 - 0.5 * h * Fx0
            J[r0:r0 + 4, 4 * (i + 1):4 * (i + 1) + 4] = I4 - 0.5 * h * Fx1
            J[r0 + 2, nX + 2 * i] = -h
            J[r0 + 3, nX + 2 * i + 1] = -h
            f0 = np.concatenate([v[i],     g[i] + U[i]])
            f1 = np.concatenate([v[i + 1], g[i + 1] + U[i]])
            J[r0:r0 + 4, -1] = -0.5 / N * (f0 + f1)
        return J

    # --------------------------------------------------------
    # Thrust magnitude limit:  |u| <= 1   (smooth, convex)
    # --------------------------------------------------------
    def thrust_limit(z):
        _, U, _ = unpack(z)
        return 1.0 - np.sum(U**2, axis=1)

    def thrust_limit_jac(z):
        _, U, _ = unpack(z)
        J = np.zeros((N, len(z)))
        i = np.arange(N)
        J[i, nX + 2 * i] = -2 * U[:, 0]
        J[i, nX + 2 * i + 1] = -2 * U[:, 1]
        return J

    grad = np.zeros(len(z0)); grad[-1] = 1.0          # objective = T
    bounds = [(None, None)] * (nX + nU) + [(1e-3, None)]

    # --------------------------------------------------------
    # Run optimization
    # --------------------------------------------------------
    with tqdm(total=max_iterations, desc="Optimizing orbit",
              unit="iteration") as pbar:

        def callback(z):
            _, U_c, T_c = unpack(z)
            pbar.update(1)
            pbar.set_postfix({
                "Time (yr)": f"{T_c * tau:.8f}",
                "Throttle avg": f"{np.mean(np.linalg.norm(U_c, axis=1)):.3f}",
                "Error": f"{np.max(np.abs(constraints(z))):.1e}",
            })

        result = minimize(
            lambda z: z[-1], z0, jac=lambda z: grad,
            method="SLSQP", bounds=bounds,
            constraints=[
                {"type": "eq",   "fun": constraints,  "jac": constraints_jac},
                {"type": "ineq", "fun": thrust_limit, "jac": thrust_limit_jac},
            ],
            callback=callback,
            options={"maxiter": max_iterations, "ftol": 1e-9, "disp": False},
        )

    print("\nOptimization finished!")
    print("Success:", result.success)
    print("Iterations:", result.nit)
    print("Function evaluations:", result.nfev)
    print("Final constraint error (scaled units):",
          np.max(np.abs(constraints(result.x))))

    # --------------------------------------------------------
    # Back to the original interface / physical units
    # --------------------------------------------------------
    X, U, T = unpack(result.x)
    X_phys = X.copy()
    X_phys[:, 2:] /= tau                      # velocity -> AU/year
    throttle = np.linalg.norm(U, axis=1)
    theta = np.arctan2(U[:, 1], U[:, 0])
    return result, X_phys, theta, throttle, T * tau


# ============================================================
# 5. VISUALIZATION
# ============================================================

def plot_solution(sys, X, theta, throttle, T):

    N = len(theta)

    times = np.linspace(0, T, N + 1)

    # Planet trajectories
    t_orbit = np.linspace(0, T, 500)

    A = np.array([
        sys.planet(t, "A")[0] for t in t_orbit
    ])

    B = np.array([
        sys.planet(t, "B")[0] for t in t_orbit
    ])

    # Planet B position and velocity
    B_states = [
        sys.planet(t, "B") for t in times
    ]

    B_pos = np.array([s[0] for s in B_states])
    B_vel = np.array([s[1] for s in B_states])

    # Rocket acceleration
    acc = np.array([
        dynamics(X[i], theta[i], throttle[i], sys)[2:]
        for i in range(N)
    ])

    # Gravity acceleration
    gravity = np.array([
        -sys.mu * X[i, :2] /
        np.linalg.norm(X[i, :2])**3
        for i in range(N)
    ])

    # Actual thrust acceleration
    thrust_acc = np.column_stack([
        sys.a * throttle * np.cos(theta),
        sys.a * throttle * np.sin(theta)
    ])

    # Rendezvous errors
    distance = np.linalg.norm(
        X[:, :2] - B_pos, axis=1
    )

    relative_velocity = np.linalg.norm(
        X[:, 2:] - B_vel, axis=1
    )

    # ========================================================
    # FIGURE 1: ORBITAL VIEW
    # ========================================================

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.scatter(0, 0, color="orange", s=250, label="Star")

    ax.plot(A[:, 0], A[:, 1], "--", label="Orbit A")
    ax.plot(B[:, 0], B[:, 1], "--", label="Orbit B")

    ax.plot(
        X[:, 0], X[:, 1],
        color="red", linewidth=2.5,
        label="Rocket trajectory"
    )

    ax.plot(
        B_pos[:, 0], B_pos[:, 1],
        color="green", alpha=0.3,
        label="Planet B trajectory"
    )

    pA, _ = sys.planet(0, "A")
    pB0, _ = sys.planet(0, "B")
    pBT, _ = sys.planet(T, "B")

    ax.scatter(*pA, color="blue", s=100, label="Planet A departure")
    ax.scatter(*pB0, color="green", s=100, label="Planet B initial")
    ax.scatter(*pBT, color="green", marker="*", s=250,
               label="Planet B arrival")

    ax.scatter(*X[0, :2], color="red", s=80)
    ax.scatter(*X[-1, :2], color="red", marker="*", s=200)

    # Thrust arrows: arrow length represents throttle
    skip = max(1, N // 25)

    for i in range(0, N, skip):

        ax.arrow(
            X[i, 0], X[i, 1],
            0.08 * throttle[i] * np.cos(theta[i]),
            0.08 * throttle[i] * np.sin(theta[i]),
            head_width=0.025,
            color="black",
            alpha=0.7
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x (AU)")
    ax.set_ylabel("y (AU)")
    ax.set_title(f"Orbital Transfer: T = {T:.5f} years")
    ax.legend()
    ax.grid()
    plt.tight_layout()

    # ========================================================
    # FIGURE 2: POSITION
    # ========================================================

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax[0].plot(times, X[:, 0], label="Rocket x")
    ax[0].plot(times, B_pos[:, 0], "--", label="Planet B x")
    ax[0].set_ylabel("x (AU)")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(times, X[:, 1], label="Rocket y")
    ax[1].plot(times, B_pos[:, 1], "--", label="Planet B y")
    ax[1].set_ylabel("y (AU)")
    ax[1].set_xlabel("Time (years)")
    ax[1].legend()
    ax[1].grid()

    fig.suptitle("Position vs Time")
    plt.tight_layout()

    # ========================================================
    # FIGURE 3: VELOCITY
    # ========================================================

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax[0].plot(times, X[:, 2], label="Rocket vx")
    ax[0].plot(times, B_vel[:, 0], "--", label="Planet B vx")
    ax[0].set_ylabel("vx (AU/year)")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(times, X[:, 3], label="Rocket vy")
    ax[1].plot(times, B_vel[:, 1], "--", label="Planet B vy")
    ax[1].set_ylabel("vy (AU/year)")
    ax[1].set_xlabel("Time (years)")
    ax[1].legend()
    ax[1].grid()

    fig.suptitle("Velocity vs Time")
    plt.tight_layout()

    # ========================================================
    # FIGURE 4: ACCELERATION
    # ========================================================

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax[0].plot(times[:-1], gravity[:, 0], label="Gravity ax")
    ax[0].plot(times[:-1], thrust_acc[:, 0], label="Thrust ax")
    ax[0].plot(times[:-1], acc[:, 0], "--", label="Total ax")
    ax[0].set_ylabel("ax (AU/year²)")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(times[:-1], gravity[:, 1], label="Gravity ay")
    ax[1].plot(times[:-1], thrust_acc[:, 1], label="Thrust ay")
    ax[1].plot(times[:-1], acc[:, 1], "--", label="Total ay")
    ax[1].set_ylabel("ay (AU/year²)")
    ax[1].set_xlabel("Time (years)")
    ax[1].legend()
    ax[1].grid()

    fig.suptitle("Acceleration vs Time")
    plt.tight_layout()

    # ========================================================
    # FIGURE 5: THRUST DIRECTION
    # ========================================================

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.step(times[:-1], np.unwrap(theta), where="post")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Thrust angle (rad)")
    ax.set_title("Thrust Direction vs Time")
    ax.grid()

    # ========================================================
    # FIGURE 6: THROTTLE
    # ========================================================

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.step(times[:-1], throttle, where="post")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Throttle")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Throttle vs Time")
    ax.grid()

    # ========================================================
    # FIGURE 7: RENDEZVOUS ERRORS
    # ========================================================

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax[0].plot(times, distance)
    ax[0].set_ylabel("Distance (AU)")
    ax[0].set_title("Distance to Planet B")
    ax[0].grid()

    ax[1].plot(times, relative_velocity)
    ax[1].set_ylabel("Relative speed (AU/year)")
    ax[1].set_xlabel("Time (years)")
    ax[1].set_title("Relative Velocity")
    ax[1].grid()

    plt.tight_layout()

    plt.show(block=True)

    # ========================================================
    # NUMERICAL SUMMARY
    # ========================================================

    print("\n========== RESULTS ==========")

    print(f"Travel time: {T:.8f} years")
    print(f"Travel time: {T * 365.25:.3f} days")

    print(f"Final position error: {distance[-1]:.8e} AU")
    print(f"Final velocity error: {relative_velocity[-1]:.8e} AU/year")

    print(f"Maximum thrust: {sys.a:.8e} AU/year^2")
    print(f"Average throttle: {np.mean(throttle):.6f}")
    print(f"Minimum throttle: {np.min(throttle):.6f}")
    print(f"Maximum throttle: {np.max(throttle):.6f}")

    print("=============================")


# ============================================================
# 6. RUN
# ============================================================

if __name__ == "__main__":

    system = System()

    print("Orbit A period:", system.periodA, "years")
    print("Orbit B period:", system.periodB, "years")

    # No T_guess here: the default (flip-and-burn estimate) is the
    # right scale. 0.5 * periodB would be ~66x too large at 1 g.
    result, X, theta, throttle, T = optimize_orbit(
        system,
        N=50
    )

    export_solution_csv(
        system,
        X,
        theta,
        throttle,
        T,
        filename="/Users/jer/Desktop/orbit_data.csv"
    )

    print("\nOptimization success:", result.success)
    print("Message:", result.message)
    print("Travel time:", T, "years")
    print("Travel time:", T * 365.25, "days")

    _, target_v = system.planet(T, "B")

    velocity_error = np.linalg.norm(
        X[-1, 2:] - target_v
    )

    print("Final velocity error:", velocity_error, "AU/year")

    plot_solution(system, X, theta, throttle, T)