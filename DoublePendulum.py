from scipy.integrate import solve_ivp
from collections     import namedtuple

import numpy                as np
import matplotlib.pyplot    as plt
import matplotlib.animation as animation


############################################################################################################
# Physics Logic
############################################################################################################

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

def compute_energy_arm_1(y, parameters):
    th1, th1d, th2, th2d = y
    l1, m1, l2, m2, g = parameters
    
    U = m1*g*l1*(1-np.cos(th1))
    
    T = 0.5*m1*(l1**2)*(th1d**2)
    
    return T, U

def compute_energy_arm_2(y, parameters):
    th1, th1d, th2, th2d = y
    l1, m1, l2, m2, g = parameters
    
    U = m2*g*(l1*(1-np.cos(th1))) + l2*(1-np.cos(th1))
    
    T = 0.5*m2*((l1**2)*(th1d**2)+2*l1*l2*th1d*th2d*np.cos(th1-th2)+(l2**2)*(th2d**2))
    
    return T, U

############################################################################################################
# Plot Generation
############################################################################################################

def make_plots(sol, parameters, t_start=None, t_end=None):
    start_idx = 0 if t_start is None else np.searchsorted(sol.t, t_start, side='left')
    end_idx = len(sol.t) if t_end is None else np.searchsorted(sol.t, t_end, side='right')
    sel = slice(start_idx, end_idx)
    
    global x1, y1, x2, y2                      # Variables for Trajectories Plot
    global t, energy, energy0, energy_scale    # Variables for Normalized Energy Error Plot
    global th1, th1d                           # Variables for Arm 1 State Space Projection
    global th2, th2d                           # Variables for Arm 2 State Space Projection 
    
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
    
    T1, U1 = compute_energy_arm_1(sol.y[:, sel], parameters)
    T2, U2 = compute_energy_arm_2(sol.y[:, sel], parameters)
    
    if(not draw_plots):
        return
    # Plot Total Energy vs. Time Graph for Arm 1
    plt.figure(dpi=120)
    plt.plot(t,T1+U1 , label="Energy Arm 1")
    plt.xlabel("Time (s)")
    plt.ylabel("Total Energy Arm 1")
    plt.title("Total Energy Arm 1 vs. Time")
    plt.grid()
    plt.show()
    # Plot Total Energy vs. Time Graph for Arm 2
    plt.figure(dpi=120)
    plt.plot(t,T2+U2 , label="Energy Arm 2")
    plt.xlabel("Time (s)")
    plt.ylabel("Total Energy Arm 2")
    plt.title("Total Energy Arm 2 vs. Time")
    plt.grid()
    plt.show()
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
    
    plt.figure(dpi=120)
    plt.plot(th1, th2, label="Arm 2")
    plt.xlabel("Theta 1 (rad)")
    plt.ylabel("Theta 2 (rad)")
    plt.title("State-Space Projection of Arm 2 in the Double Pendulum System")
    plt.grid()
    plt.show()
    
    '''
    plt.figure(dpi=120)
    plt.plot(th2, th2d, label="Arm 2", color='RED')
    plt.plot(th2, th2d, label="Arm 2")
    plt.xlabel("Theta 2 (rad)")
    plt.ylabel("Theta 2 dot (rad/s)")
    plt.title("State-Space Projection of Arm 2 in the Double Pendulum System")
    plt.grid()
    plt.show()
    '''
    
    


############################################################################################################
# Plot Setup
############################################################################################################
    
# Whether or not to display the static images of the plots.
draw_plots = True

parameters = Parameters(
    l1 = 0.5,
    m1 = 1000,
    l2 = 0.5,
    m2 = 1.0,
    g = 9.8,
)

'''init_conditions = DepVariables(
    th1 = 2.899,
    th1d = 0,
    th2 = 1.914,
    th2d = 0,
)
'''
#Initial Conditions for Choatic/High Energy System Exploration: 
init_conditions = DepVariables(
    th1 = 2.67,
    th1d = 0,
    th2 = 3.12,
    th2d = 0,
 )
t_span = (0.0, 20.0)
# These are output sampling times; solve_ivp chooses adaptive internal timesteps.
num_points = 1001
t_eval = np.linspace(*t_span, num_points)

sol = solve_ivp(
    eom_double_pendulum, t_span, init_conditions, args=parameters,
    rtol=1e-7, atol=1e-14, t_eval=t_eval, method="DOP853"
)
print(sol.message)

make_plots(sol, parameters, t_start=None, t_end=None)



############################################################################################################
# Animation Generation
############################################################################################################

### Animation config variables
make_traj_animation = False
make_energy_animation = False
make_state_space_animation_1 = False
make_state_space_animation_2 = False

repeat_animation = True

total_duration      = 5  # Total Duration of the animation in [s]
loop_delay          = 100 # Delay in gif restart in [ms]
step_size_per_frame = 5  # How many points to skip per frame, for animation generation performace

# Total duration[ms] = number of frames[#] * frame interval[ms]
num_frames     = (num_points + step_size_per_frame - 1) // step_size_per_frame
frame_interval = (total_duration*1e3) // num_frames

# Framerate for gif export
fps = num_frames/total_duration  


### Animation frame generation functions
# Trajectories animation function
def updateTraj(frame):
    frame_index = (frame) * step_size_per_frame
    
    x1_slice = x1[:frame_index]
    y1_slice = y1[:frame_index]
    
    x2_slice = x2[:frame_index]
    y2_slice = y2[:frame_index]
    
    lines1[0].set_data(x1_slice, y1_slice)
    lines1[1].set_data(x2_slice, y2_slice)
    
    if frame_index > 0:
        points1[0].set_data([x1_slice[-1]], [y1_slice[-1]])
        points1[1].set_data([x2_slice[-1]], [y2_slice[-1]])
    
    return tuple(lines1 + points1)

# Energy error function
def updateEnergyError(frame):
    frame_index = (frame) * step_size_per_frame
    
    t_slice = t[:frame_index]
    e_slice: float = (energy[:frame_index] - energy0) / energy_scale

    line2.set_data(t_slice, e_slice)
    
    if frame_index > 0:
        point2.set_data([t_slice[-1]], [e_slice[-1]])

    return line2, point2,

# State-Space arm 1 function
def updateStateSpace_1(frame):
    frame_index = (frame) * step_size_per_frame
    
    th1_slice = th1[:frame_index]
    th1d_slice = th1d[:frame_index]

    line3.set_data(th1_slice, th1d_slice)
    
    if frame_index > 0:
        point3.set_data([th1_slice[-1]], [th1d_slice[-1]])

    return line3, point3,

# State-Space arm 2 function
def updateStateSpace_2(frame):
    frame_index = (frame) * step_size_per_frame
    
    th2_slice = th2[:frame_index]
    th2d_slice = th2d[:frame_index]

    line4.set_data(th2_slice, th2d_slice)
    
    if frame_index > 0:
        point4.set_data([th2_slice[-1]], [th2d_slice[-1]])

    return line4, point4,

### Animation plots
# Trajectories animation plot
if(make_traj_animation):
    ani_fig1, ax1 = plt.subplots(dpi=120)
    line1_1,  = ax1.plot([],[], lw=2, color='blue', label="Mass 1")
    point1_1, = ax1.plot([],[], 'o', color='blue', markersize=4)
    line1_2,  = ax1.plot([],[], lw=2, color='orange', label="Mass 2")
    point1_2, = ax1.plot([],[], 'o', color='orange', markersize=4)
    
    lines1 = [line1_1, line1_2]
    points1 = [point1_1, point1_2]
    
    ax1.plot([0], [0], "ko")  # origin
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("Trajectories of the Masses in the Double Pendulum System")
    ax1.legend()
    ax1.set_xlim(-0.45,0.45)
    ax1.set_ylim(-1.1,0.05)
    ax1.grid()
    
    ani1 = animation.FuncAnimation(ani_fig1, updateTraj, frames=num_frames, interval=frame_interval, blit=True, repeat=repeat_animation, repeat_delay=loop_delay)
    save_path = "animations/TrajPlot.gif"
    ani1.save(save_path, writer='pillow', fps=fps)


# Trajectories animation plot
if(make_energy_animation):
    ani_fig2, ax2 = plt.subplots(dpi=120)
    line2,  = ax2.plot([],[], lw=2, color='purple', label="Normalized energy error")
    point2, = ax2.plot([],[], 'o', color='purple', markersize=4)
    
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("(Energy - Initial Energy) / Energy Scale")
    ax2.set_title("Normalized Energy Error of the Double Pendulum System")
    ax2.set_xlim(-0.3,5.3)
    ax2.set_ylim(-0.6e-7,4e-7)
    ax2.grid()
    
    ani2 = animation.FuncAnimation(ani_fig2, updateEnergyError, frames=num_frames, interval=frame_interval, blit=True, repeat=repeat_animation, repeat_delay=loop_delay)
    save_path = "animations/EnergyPlot.gif"
    ani2.save(save_path, writer='pillow', fps=fps)

# State-space projection arm 1 plot
if(make_state_space_animation_1):
    ani_fig3, ax3 = plt.subplots(dpi=120)
    line3,  = ax3.plot([],[], lw=2, color='blue', label="Arm 1")
    point3, = ax3.plot([],[], 'o', color='blue', markersize=4)
    
    ax3.set_xlabel("Theta 1 (rad)")
    ax3.set_ylabel("Theta 1 dot (rad/s)")
    ax3.set_title("State-Space Projection of Arm 1 in the Double Pendulum System")
    ax3.legend()
    ax3.set_xlim(-0.7,0.6)
    ax3.set_ylim(-4,3.5)
    ax3.grid()
    
    ani3 = animation.FuncAnimation(ani_fig3, updateStateSpace_1, frames=num_frames, interval=frame_interval, blit=True, repeat=repeat_animation, repeat_delay=loop_delay)
    save_path = "animations/StateSpacePlot_1.gif"
    ani3.save(save_path, writer='pillow', fps=fps)
    
# State-space projection arm 2 plot
if(make_state_space_animation_2):
    ani_fig4, ax4 = plt.subplots(dpi=120)
    line4,  = ax4.plot([],[], lw=2, color='blue', label="Arm 2")
    point4, = ax4.plot([],[], 'o', color='blue', markersize=4)
    
    ax4.set_xlabel("Theta 2 (rad)")
    ax4.set_ylabel("Theta 2 dot (rad/s)")
    ax4.set_title("State-Space Projection of Arm 2 in the Double Pendulum System")
    ax4.legend()
    ax4.set_xlim(-1,1)
    ax4.set_ylim(-5.5,5.5)
    ax4.grid()
    
    ani4 = animation.FuncAnimation(ani_fig4, updateStateSpace_2, frames=num_frames, interval=frame_interval, blit=True, repeat=repeat_animation, repeat_delay=loop_delay)
    save_path = "animations/StateSpacePlot_2.gif"
    ani4.save(save_path, writer='pillow', fps=fps)