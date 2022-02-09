##############################################################################
# This piece of code was written to simulate n astronomical bodies. The code
# has been configured as per the investigation I was performing for the
# assignment: the study of the effect of a rogue flyby star.
##############################################################################

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Constants
G = 6.674e-11
year = 3.27e7

# Setup conditions
step_n = 100000
tmax = 20*year
animate = False
solar_centring = True

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

# Calculates the distance between two bodies at some time
def distance(data, time, i, j):
    return np.sqrt((data[time,4*i]-data[time,4*j])**2 + (data[time,4*i+1]-data[time,4*j+1])**2)

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

# Centres the input data around a particular body index
def centre(data, c_index):
    global system
    global N    
    
    for i in range(N):
        if i != c_index:
            data[:, 4*i] -= data[:,4*c_index]
            data[:,4*i+1] -= data[:,4*c_index+1]
            
    data[:,4*c_index] = 0
    data[:,4*c_index+1] = 0
    
    return data

# Sets up the specific celestial bodies using the body class
moon = body(7.34767309e22, 1737100, (147.47e9 + 384.4e6), 0, 0, 30.29e3+1.08e3, "Moon", 'grey')
earth = body(5.972e24, 6371000, 147.47e9, 0, 0, 30.29e3, "Earth", 'blue')
sun = body(1.989e30, 696340000, 0, 0, 0, 0, "Sun", 'orange')
mercury = body(0.33011e24, 2439.7e3, 46e9, 0, 0, 58.98e3, "Mercury", 'salmon')
venus = body(4.8675e24, 6051.8e3, 107.476e9, 0, 0, 35.26e3, "Venus", 'cyan')
mars = body(0.64171e24, 3396.2e3, 206.617e9, 0, 0, 26.50e3, "Mars", 'red')
sunf = body(1.989e30*0.1, 696340000, -1*3.4e11, -3.4e11, 26.5e3, 53e3, "Star Flyby", 'yellow')

system = [sun, earth, moon, sunf]

# Sets up an array to provide a step of x starting positions for the flyby star
distance_step = [1.1, 1.08, 1.06, 1.04, 1.02, 1]

# Sets up the figures and axes
fig_d, ax_d = plt.subplots(len(distance_step), figsize=(16,8))
fig, ax = plt.subplots(2, len(distance_step), figsize=(16,8))

# Loop which runs the n-body simulation
for test in range(len(distance_step)):
    
    # Multiplies the initial x coordinate of the flyby star by the corresponding step array element
    if sunf in system:
        system[system.index(sunf)].x0 = -3.4e11*distance_step[test]
        

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
    
    # Sets up the time step
    t = np.linspace(0, tmax, step_n)
    k_e = [0]*len(t)
    p_e = [0]*len(t)
    energy = [0]*len(t)
    
    # Calculates the orbits by differentiating the ode function
    orbits = []
    orbits = odeint(ode, data, t)
    
    # Plots the orbits of the system and x-position against time for Earth
    for i in range(N):
        if solar_centring:
            pltorbits = centre(orbits, system.index(sun))
            ax[0, test].plot(pltorbits[:,4*i], pltorbits[:,4*i+1], color=system[i].colour)
            ax[0, test].plot(0,0,'.', color=system[system.index(sun)].colour)
        else:
            ax[0, test].plot(orbits[:,4*i], orbits[:,4*i+1], color=system[i].colour)
        
        
        # Sets axis limits to within 600Gm from the Sun
        ax[0, test].set_xlim(orbits[0, 4*system.index(sun)]-600e9, orbits[0, 4*system.index(sun)]+600e9)
        ax[0, test].set_ylim(orbits[0, 4*system.index(sun)+1]-600e9, orbits[0, 4*system.index(sun)+1]+600e9)
        
        ax[0, test].set_xlabel("x/m")
        ax[0, test].set_ylabel("y/m")
       
    # Places a label on the first orbit plot
    orbithandle = []
    for i in range(N):
        orbithandle.append(mpatches.Patch(color=system[i].colour, label=system[i].name))
    ax[0,0].legend(handles=orbithandle)
        
    
    # Creates a subplot containing the x-position of the Earth against time, to show whether or not it's nature is oscillatory
    if earth in system:
        ax[1,test].plot(t, orbits[:,4*system.index(earth)], 'blue')
        
        ax[1,test].title.set_text("Earth")
        
        ax[1,0].set_ylabel("x/m")
        ax[1,test].set_xlabel("t/s")
    
    
    if sun in system and earth in system and sunf in system:
        # Calculates the distance between the Earth and the Sun/Flyby Star at every time point
        distances = [0]*len(t)
        distancesfb = [0]*len(t)
        for a in range(len(t)):
            distances[a] = distance(orbits, a, system.index(earth), system.index(sun))
            distancesfb[a] = distance(orbits, a, system.index(earth), system.index(sunf))
        
        # Plots graphs of Earth-Sun and Earth-Flyby distances against time for each Flyby initial x-position step. Labels the graphs with
        # the gradients of the Earth-Sun line to see if the Earth is moving away from the sun.
        ax_d[test].plot(t, distances, 'blue')
        ax_d[test].plot(t, distancesfb, 'red')
        ax_d[test].set_xlabel("t/s")
        
        slope, intercept = np.polyfit(t, distances, 1)
        
        ax_d[test].set_title("Gradient of earth-sun distance = " + str(slope))
        ax_d[round(len(distance_step)/2)].set_ylabel("Distance between the Earth and Sun and Star/m")
        
        
        # Labels the orbit plots with the minimum Earth-Flyby Star distance, and plot the positions at which they are at their closest
        # with red 'x's.
        for i in range(N):
            ax[0, test].set_title("Distance = " + str(round(min(distancesfb)/1.469e11, 4)) + "AU", y=1.08)
        
        ax[0, test].plot(orbits[distancesfb.index(min(distancesfb)), 4*system.index(earth)], orbits[distancesfb.index(min(distancesfb)), 4*system.index(earth)+1], 'x', color='red')
        ax[0, test].plot(orbits[distancesfb.index(min(distancesfb)), 4*system.index(sunf)], orbits[distancesfb.index(min(distancesfb)), 4*system.index(sunf)+1], 'x', color='red')
        
        line_esf = mpatches.Patch(color='red', label='Earth-Flyby Distance')
        line_es = mpatches.Patch(color='blue', label='Earth-Sun Distance')
        ax_d[0].legend(handles=[line_es, line_esf])

# Adjusts figures to allow axis labels to be properly visible
fig.tight_layout()
fig_d.subplots_adjust(hspace=1.2)