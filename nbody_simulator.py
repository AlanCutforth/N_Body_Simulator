##############################################################################
# This piece of code was written to simulate n astronomical bodies. The code
# has been configured as per the assignment to replicate the behaviour of the
# solar system, however it can be configured to simulate any astronomical body
# system.
##############################################################################

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Constants
G = 6.674e-11
year = 3.27e7

# Setup conditions
step_n = 10000
tmax = 2*year
animate = True


# Sets up a class with which to store data about celestial bodies
class body:
    def __init__(self, _mass, _radius, _x0, _y0, _vx0, _vy0, _name, _colour):
        self.mass = _mass
        self.radius = _radius
        self.x0 = _x0
        self.y0 = _y0
        self.vx0 = _vx0
        self.vy0 = _vy0
        self.name = _name
        self.colour = _colour

# Calculates the x and y acceleration induced on body i by body j
def acceleration(xi, yi, xj, yj, mj):
    rsqr = (xi-xj)**2 + (yi-yj)**2
    x_a = -G*mj*(xi - xj)/(rsqr**1.5)
    y_a = -G*mj*(yi - yj)/(rsqr**1.5)
    
    return [x_a, y_a]

# Calculates kinetic energy
def kinetic(vxi, vyi, mi):
    return 0.5*mi*(vxi**2 + vyi**2)

# Calculates the potential energy between bodies i and j
def potential(xi, yi, mi, xj, yj, mj):
    r = np.sqrt((xi-xj)**2 + (yi-yj)**2)
    return -G*mj*mi/r

# Differential function    
def ode(data, t):
    
    diff = [np.NaN]*4*N
    for i in range(N):
        ax = 0
        ay = 0
        diff[4*i] = data[4*i+2] # Sets the "position" index for a body i to the current x velocity ready to be differentiated to position 
        diff[4*i+1] = data[4*i+3] # Sets the "position" index for a body i to the current y velocity ready to be differentiated to position 
        for j in range(N): # Sums the accelerations due to the forces acting on body i from all bodies j
            if i != j: # Ignores the effect of body i on itself
                ax += acceleration(data[4*i], data[4*i+1], data[4*j], data[4*j+1], system[j].mass)[0]
                ay += acceleration(data[4*i], data[4*i+1], data[4*j], data[4*j+1], system[j].mass)[1]
        
        diff[4*i+2] = ax # Sets the "velocity" index for a body i to the x acceleration ready to be differentiated to x velocity
        diff[4*i+3] = ay # Sets the "velocity" index for a body i to the y acceleration ready to be differentiated to y velocity
    return diff

# Sets up orbit axis labels and legend
def label_orbits(fig, ax):
    global system
    
    ax.set_xlabel("x/m")
    ax.set_ylabel("y/m")
    ax.title.set_text("Orbits")
    orbithandle = []
    for i in range(N):
        orbithandle.append(mpatches.Patch(color=system[i].colour, label=system[i].name))
    ax.legend(handles=orbithandle)

# Sets up the specific celestial bodies using the body class
moon = body(7.34767309e22, 1737100, (147.47e9 + 384.4e6), 0, 0, 30.29e3+1.08e3, "Moon", 'grey')
earth = body(5.972e24, 6371000, 147.47e9, 0, 0, 30.29e3, "Earth", 'blue')
sun = body(1.989e30, 696340000, 0, 0, 0, 0, "Sun", 'orange')
mercury = body(0.33011e24, 2439.7e3, 46e9, 0, 0, 58.98e3, "Mercury", 'salmon')
venus = body(4.8675e24, 6051.8e3, 107.476e9, 0, 0, 35.26e3, "Venus", 'cyan')
mars = body(0.64171e24, 3396.2e3, 206.617e9, 0, 0, 26.50e3, "Mars", 'red')

system = [sun, earth, moon, mercury, venus, mars]

data = []
N = len(system) # Number of bodies

# Calculates the centre of mass of the entire system and offsets the positioning of the system such that the
# centre of mass is the centre of the system. This prevents the entire system wiggling off course as time passes
cm_x = 0
cm_vx = 0
cm_y = 0
cm_vy = 0
masstot = 0

for i in range(N):
    cm_x += system[i].x0*system[i].mass
    cm_vx += system[i].vx0*system[i].mass
    cm_y += system[i].y0*system[i].mass
    cm_vy += system[i].vy0*system[i].mass
    masstot += system[i].mass
    
cm_x /= masstot
cm_vx /= masstot
cm_y /= masstot
cm_vy /= masstot

# Sets up the initial conditions for the data array
for i in range(N):
    data.append(system[i].x0 - cm_x)
    data.append(system[i].y0 - cm_y)
    data.append(system[i].vx0 - cm_vx)
    data.append(system[i].vy0 - cm_vy)

# Sets up the time step and blank energy arrays
t = np.linspace(0, tmax, step_n)
k_e = [0]*len(t)
p_e = [0]*len(t)
energy = [0]*len(t)

# Calculates the orbits by differentiating the ode function
orbits = odeint(ode, data, t)

# Plots the orbits for each body
fig = plt.figure(figsize=(10,8))
ax = plt.axes()
label_orbits(fig, ax)

# Plots the obrits for the system, after first checking if they are to be animated
if animate == True:
    for a in range(len(t)):
        if a%10 == 0:
            ax.cla()
            for i in range(N):
                ax.plot(orbits[:,4*i], orbits[:,4*i+1], color=system[i].colour)
                ax.plot(orbits[a,4*i], orbits[a,4*i+1], color=system[i].colour, marker='o', markersize=4)
                
                label_orbits(fig, ax)
                
            plt.pause(0.0001)
else:
    for i in range(N):
        ax.plot(orbits[:,4*i], orbits[:,4*i+1], color=system[i].colour)

# Sums the energy of the entire system and divides by the first value to make sure that the total
# energy remains constant. Potential is halved to take the average potential energy for each body-body system
for a in range(len(t)):
    for i in range(N):
        k_e[a] += kinetic(orbits[a,4*i+2], orbits[a,4*i+3], system[i].mass)
        for j in range(N):
            if i != j:
                p_e[a] += potential(orbits[a,4*i], orbits[a,4*i+1], system[i].mass, orbits[a,4*j], orbits[a,4*j+1], system[j].mass)/2
    energy[a] += k_e[a] + p_e[a]


# Plots the graph of the energies of the system over time
fig_e = plt.figure(figsize=(12,7))
ax_e = plt.axes()
ax_e.plot(t, energy, 'black')
ax_e.plot(t, [energy[0]]*len(t), 'red')
ax_e.plot(t, k_e, 'blue')
ax_e.plot(t, p_e, 'purple')
ax_e.set_xlabel("t/s")
ax_e.set_ylabel("Energy/J")
e_diff = max(energy) - min(energy)
ax_e.title.set_text("Energy range = " + str(e_diff) + "J. Ratio to initial energy is " + str(np.abs(e_diff/energy[0])) + ".")
line = mpatches.Patch(color='red', label='Initial System Energy')
line_init = mpatches.Patch(color='black', label='System Energy')
line_k = mpatches.Patch(color='blue', label='Kinetic Energies')
line_p = mpatches.Patch(color='purple', label='Potential Energies')
ax_e.legend(handles=[line, line_init, line_k, line_p])

# Creates a subplot containing the x and y values of every body over time, to show their oscillatory nature and prove the system is true n-body
figxy, axxy = plt.subplots(2,N, figsize=((3*N),6))
for i in range(N):
    axxy[0,i].plot(t, orbits[:,4*i], 'blue')
    axxy[1,i].plot(t, orbits[:,4*i+1], 'red')
    
    axxy[0,i].title.set_text(system[i].name)
    
    axxy[0,0].set_ylabel("x/m")
    axxy[1,0].set_ylabel("y/m")
    axxy[1,i].set_xlabel("t/s")

# According to the Virial Theorem, 2(Kinetic Energy) + Potential = 0 for bound orbits. This calculates the ratio of this "0" value to the initial energy,
# to show whether or not it is close to 0.
virial_r = [0]*len(t)
for a in range(len(t)):
    virial_r[a] = 2*k_e[a] + p_e[a]
    virial_r[a] /= energy[0]
    
fig_v = plt.figure()
ax_v = plt.axes()
ax_v.plot(t, virial_r, 'cyan')
ax_v.set_xlabel("t/s")
ax_v.set_ylabel("(2K + U)/Initial Energy")
ax_v.title.set_text("Virial Theorem")
ax_v.plot(t, [0]*len(t), 'red')