import numpy as np
import copy
import sys
import random
import time
import matplotlib.pyplot as plt
# def fitness(particle , problem):

#     fit = 0.0
#     if problem == 1: # rosenbrock
#         for j in range(len(particle)-1):
#             fit += 100.0*((particle[j]**2 - particle[j+1])**2) + (particle[j]-1.0)**2
#     elif problem ==2:  # ellipsoidal - sphere function
#         for j in range(len(particle)):
#             fit += (j+1)*(particle[j]**2)
#     elif problem ==3:  # rastrigin's function
#         fit = 10*len(particle)
#         for j in range(len(particle)):
#             fit += (particle[j]**2 - 10*math.cos(2*math.pi*particle[j]))
#     elif problem ==4:  # ackeley function
#         fit = -20*np.exp(-0.2*np.sqrt(0.5*(particle[0]**2+particle[1]**2))) - np.exp(0.5*(math.cos(2*math.pi*particle[0]) + math.cos(2*math.pi*particle[1]))) + math.e +20 
#     elif problem ==5: #eggholder function
#         fit = -(particle[1]+47)*math.sin(np.sqrt(abs(particle[1]+(particle[0]/2)+47))) - particle[0]*math.sin(np.sqrt(abs(particle[0]-particle[1]-47)))
#     elif problem ==6: #easom function
#         fit = -math.cos(particle[0])*math.cos(particle[1])*np.exp(-((particle[0]-math.pi)**2 + (particle[1]-math.pi)**2 ))            
#     elif problem ==7: #styblinski-tang function
#         for j in range(len(particle)):
#             fit = fit + (particle[j]**4 - 16*(particle[j]**2) + 5*particle[j])/2   


class EA:
    
    def __init__(self,pop_size,dim,bounds,problem,num_evals):

        self.pop_size = pop_size
        self.dim = dim
        self.bounds = bounds
        self.position = np.zeros((self.pop_size,self.dim))
        self.velocity = np.zeros((self.pop_size,self.dim))
        self.p_best = copy.copy(self.position)
        self.g_best = sys.float_info.max
        self.num_evals = num_evals

        self.problem = problem
            
    
        for i in range(pop_size):
            np_pos = np.random.rand(dim)
            np_vel = np.random.rand(dim)
        
            self.position[i] = list((bounds[1] - bounds[0]) * np_pos  + bounds[0]) 
            self.velocity[i] = list((bounds[1] - bounds[0]) * np_vel  + bounds[0])

        self.p_best_score = self.evaluate()

    def evaluate(self):
        swarm = self.position
        problem = self.problem
        if problem ==2:  # ellipsoidal - sphere function
            matrix = np.array([np.arange(1,self.dim+1),]*self.pop_size)
            fit =((swarm**2)*matrix)
            fit = np.sum(fit , axis=1)
            return fit

    def run(self):
        # ab hoga asla
        evals=0
        score_list = []
        eval_list = []
        while evals<self.num_evals:
            score = self.evaluate()

            for i in range(self.pop_size):

                if score[i]<self.p_best_score[i]:
                    self.p_best[i]=copy.copy(self.position[i])
                    self.p_best_score[i] = score[i]

                if score[i]<self.g_best:
                    self.g_best = score[i]
                    g_best_pos = copy.copy(self.position[i])
        
            r1 = np.random.rand(self.dim)
            r2 = np.random.rand(self.dim)
            w=0.729       # constant inertia weight (how much to weigh the previous velocity)
            c1=1.4        # cognative constant
            c2=1.4        # social constant
            
            vel_cognitive=c1*r1*(self.p_best-self.position)
            vel_social=c2*r2*(g_best_pos-self.position)
            self.velocity=w*self.velocity+vel_cognitive+vel_social
            self.position += self.velocity

            print("Best Score:",self.g_best)
            print("At:",g_best_pos)

            evals+=self.pop_size
            score_list.append(self.g_best)
            eval_list.append(evals)
            if len(score_list)>100:
                if np.all(score_list[-20:][0]==score_list[-10:]):
                    print("Stopping due to probable saturation at evals:",evals)
                    break

        plt.plot(eval_list,score_list)
        plt.show()


if __name__ == "__main__":
    start= time.time()
    a = EA(args) # put those args there
    a.run()
    print("Time Taken = ",(time.time()-start)/60 ,"Minutes")
