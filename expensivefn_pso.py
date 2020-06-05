import random
import numpy as np 
import matplotlib.pyplot as plt
import math
W = 0.5
c1 = 0.8
c2 = 0.9

n_iterations = 100
target_error = 0
n_particles = 1000
dim= 5
class Particle():
    def __init__(self):
        self.position = np.zeros((dim,))
        for i in range(dim):
            self.position[i] = (-1) ** (bool(random.getrandbits(1))) * random.random()*50
        self.pbest_position = self.position
        self.pbest_value = float('inf')
        self.velocity = np.zeros((dim , ))

    def __str__(self):
        print("I am at ", self.position, " meu pbest is ", self.pbest_position)
    
    def move(self):
        self.position = self.position + self.velocity


class Space():

    def __init__(self, target, target_error, n_particles, problem):
        self.target = target
        self.target_error = target_error
        self.n_particles = n_particles
        self.particles = []
        self.gbest_value = float('inf')
        self.gbest_position = np.array([random.random()*50, random.random()*50])
        self.problem = problem
    def print_particles(self):
        for particle in self.particles:
            particle.__str__()
   
    def fitness(self, particle):
        
        fit = 0.0
        if self.problem == 1: # rosenbrock
            for j in range(dim):
                fit += (100.0*(particle.position[j]*particle.position[j] - particle.position[j+1])*(particle.position[j]*particle.positionx[j] - particle.position[j+1]) + (particle.position[j]-1.0)*(particle.position[j]-1.0))
        elif self.problem ==2:  # ellipsoidal - sphere function
            for j in range(dim):
                fit = fit + ((j+1)*(particle.position[j]*particle.position[j]))
        elif self.problem ==3:  # rastrigin's function
            fit = 10*dim
            for j in range(dim):
                fit = fit + (particle.position[j]*particle.position[j] - 10*math.cos(2*math.pi*particle.position[j]))
        elif self.problem ==4:  # ackeley function
            fit = -20*np.exp(-0.2*np.sqrt(0.5*(particle.position[0]*particle.position[0]+particle.positionx[1]*particle.position[1]))) - np.exp(0.5*(math.cos(2*math.pi*particle.position[0]) + math.cos(2*math.pi*particle.position[1]))) + math.e +20 
        elif self.problem ==5: #eggholder function
            fit = -(x[1]+47)*math.sin(np.sqrt(abs(particle.position[1]+(particle.position[0]/2)+47))) - x[0]*math.sin(np.sqrt(abs(particle.position[0]-particle.position[1]-47)))
        elif self.problem ==6: #easom function
            fit = -math.cos(particle.position[0])*math.cos(particle.position[1])*np.exp(-((particle.position[0]-math.pi)**2 + (particle.position[1]-math.pi)**2 ))            
        elif self.problem ==7: #styblinski-tang function
            for j in range(dim):
                fit = fit + (particle.position[j]**4 - 16*(particle.position[j]**2) + 5*particle.position[j])/2   
            
        return fit # note we will maximize fitness, hence minimize error

    
    def set_pbest(self):
        for particle in self.particles:
            fitness_cadidate = self.fitness(particle)
            if(particle.pbest_value > fitness_cadidate):
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position
            

    def set_gbest(self):
        for particle in self.particles:
            best_fitness_cadidate = self.fitness(particle)
            if(self.gbest_value > best_fitness_cadidate):
                self.gbest_value = best_fitness_cadidate
                self.gbest_position = particle.position

    def move_particles(self):
        for particle in self.particles:
            global W
            new_velocity = (W*particle.velocity) + (c1*random.random()) * (particle.pbest_position - particle.position) + \
                            (random.random()*c2) * (self.gbest_position - particle.position)
            particle.velocity = new_velocity
            particle.move()
            

search_space = Space(1, target_error, n_particles , 2)
particles_vector = [Particle() for _ in range(search_space.n_particles)]
search_space.particles = particles_vector
search_space.print_particles()
x=[]
y=[]
ite = []
iteration = 0
while(iteration < n_iterations):
    search_space.set_pbest()    
    search_space.set_gbest()

    if(abs(search_space.gbest_value - search_space.target) <= search_space.target_error):
        break

    search_space.move_particles()
    iteration += 1
        
    x.append(search_space.gbest_position[0])
    y.append(search_space.gbest_position[1])
    ite.append(iteration)

plt.plot(ite,x)
plt.plot(ite,y)

plt.show()
print("The best solution is: ", search_space.gbest_position,"Value" ,search_space.gbest_value,  " in n_iterations: ", iteration)
