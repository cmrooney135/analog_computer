import matplotlib.pyplot as plt 
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk


#these will be read in from the mcu 
ALPHA = 0.1  
BETA = 0.01
GAMMA = 0.2 
DELTA = 0.02 

# x is prey population
# y is predator population 
# alpha is growth rate of prey
# beta is predator-prey interaction rate
# gamma is predator death rate
# delta is predator reproductive rate 


#function for differential equation 
def dydt(t, y):
    x, z = y
    dxdt = ALPHA * x - BETA * x * z  # Prey equation
    dzdt = DELTA * x * z - GAMMA * z  # Predator equation
    
    return [dxdt, dzdt]

#initial conditions 
y_0 = [50, 10]  


#timespan
timespan = (0, 100)

#use ivp to solve equation 
solution = solve_ivp(dydt, timespan, y_0, method='RK45', t_eval=np.linspace(0, 100, 500))

'''
plt.plot(solution.t, solution.y[0], label='Prey (x)')
plt.plot(solution.t, solution.y[1], label='Predator (z)')
plt.xlabel('Time t')
plt.ylabel('Population')
plt.title("Predator-Prey Model")
plt.legend()
plt.show()
'''
fig, ax = plt.subplots()
ax.set_xlim(0, 100)  
ax.set_ylim(0, max(y_0) * 2)  

line_prey, = ax.plot([], [], label='Prey (x)', color='blue')
line_predator, = ax.plot([], [], label='Predator (z)', color='red')

ax.set_xlabel('Time t')
ax.set_ylabel('Population')
ax.set_title("Predator-Prey Model")
ax.legend()

def init():
    line_prey.set_data([], [])
    line_predator.set_data([], [])
    return line_prey, line_predator

def update(frame):
    line_prey.set_data(solution.t[:frame], solution.y[0][:frame])  
    line_predator.set_data(solution.t[:frame], solution.y[1][:frame])  
    return line_prey, line_predator

ani = FuncAnimation(fig, update, frames=len(solution.t), init_func=init, blit=True, interval=50)

ani.save('Predator_prey0.gif', writer='pillow')


plt.show()