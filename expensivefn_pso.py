from __future__ import division
from random import random
from random import uniform
import numpy as np
import matplotlib.pyplot as plt
import math

class Particle:
    def __init__(self,num_dimensions,bounds):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.num_dimensions=num_dimensions
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(uniform(-1,1))
            p = random()
            n = p*bounds 
            x = random()
            if x>0.5:
                self.position_i.append(n[0])
            else:
                self.position_i.append(n[1])

    # evaluate current fitness
    def evaluate(self,costFunc,problem):
        self.err_i=costFunc(self.position_i , problem)

        # check to see if the current position is an individual best
        if self.err_i<self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i.copy()
            self.err_best_i=self.err_i
                    
    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.729       # constant inertia weight (how much to weigh the previous velocity)
        c1=1.4        # cognative constant
        c2=1.4        # social constant
        
        for i in range(0,self.num_dimensions):
            r1=random()
            r2=random()
            
            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,self.num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]
            
            # adjust maximum position if necessary
            if self.position_i[i]>bounds[1]:
                self.position_i[i]=bounds[1]

            # adjust minimum position if neseccary
            if self.position_i[i]<bounds[0]:
                self.position_i[i]=bounds[0]
        
        
def minimize(costFunc,num_dimensions, bounds, num_particles, maxiter, problem ,verbose=False):
#     global num_dimensions

    err_best_g=-1                   # best error for group
    pos_best_g=[]                   # best position for group

    # establish the swarm
    swarm=[]
    for i in range(0,num_particles):
        swarm.append(Particle(num_dimensions , bounds))

    # begin optimization loop
    i=0
    a = []
    v = [[] for h in range(num_dimensions)]
    ite = []
    while i<maxiter:
        if verbose: print(f'iter: {i:>4d}, best solution: {err_best_g:10.6f}')
       
    
        a.append(err_best_g)
        ite.append(i)
            
        # cycle through particles in swarm and evaluate fitness
        for j in range(0,num_particles):
            swarm[j].evaluate(costFunc , problem)

            # determine if current particle is the best (globally)
            if swarm[j].err_i<err_best_g or err_best_g==-1:
                pos_best_g=list(swarm[j].position_i)
                err_best_g=float(swarm[j].err_i)
        
        for k in range(num_dimensions):
            v[k].append(pos_best_g[k])
        
        # cycle through swarm and update velocities and position
        for j in range(0,num_particles):
            swarm[j].update_velocity(pos_best_g)
            swarm[j].update_position(bounds)
            
        i+=1
    
    plt.plot(ite,a)
    plt.show()

    # print final results
    print('\nFINAL SOLUTION:')
    print(f'   > {pos_best_g}')
    print(f'   > {err_best_g}\n')
    
    for k in range(num_dimensions):
        plt.plot(ite,v[k])
    plt.show
    
def fitness(particle , problem):

    fit = 0.0
    if problem == 1: # rosenbrock
        for j in range(len(particle)-1):
            fit += 100.0*((particle[j]**2 - particle[j+1])**2) + (particle[j]-1.0)**2
    elif problem ==2:  # ellipsoidal - sphere function
        for j in range(len(particle)):
            fit += (j+1)*(particle[j]**2)
    elif problem ==3:  # rastrigin's function
        fit = 10*len(particle)
        for j in range(len(particle)):
            fit += (particle[j]**2 - 10*math.cos(2*math.pi*particle[j]))
    elif problem ==4:  # ackeley function
        fit = -20*np.exp(-0.2*np.sqrt(0.5*(particle[0]**2+particle[1]**2))) - np.exp(0.5*(math.cos(2*math.pi*particle[0]) + math.cos(2*math.pi*particle[1]))) + math.e +20 
    elif problem ==5: #eggholder function
        fit = -(particle[1]+47)*math.sin(np.sqrt(abs(particle[1]+(particle[0]/2)+47))) - particle[0]*math.sin(np.sqrt(abs(particle[0]-particle[1]-47)))
    elif problem ==6: #easom function
        fit = -math.cos(particle[0])*math.cos(particle[1])*np.exp(-((particle[0]-math.pi)**2 + (particle[1]-math.pi)**2 ))            
    elif problem ==7: #styblinski-tang function
        for j in range(len(particle)):
            fit = fit + (particle[j]**4 - 16*(particle[j]**2) + 5*particle[j])/2   

    return fit # note we will maximize fitness, hence minimize error
