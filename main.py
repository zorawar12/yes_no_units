# Nature of collective-decision making by simple 
# yes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin12a@iiserb.ac.in

#%%
# Libraries___________________________________________
import numpy as np
import matplotlib.pyplot as plt

# Functions___________________________________________

def dm(population_size,number_of_options):
    sum = population_size+1

    while sum>population_size:
        Dm = np.random.normal(mum,sigmam,number_of_options)
        sum = 0
        for m in Dm:
            sum += int(m)
    return Dm

def units_assign_to_opt(Dh,Dm,number_of_options):
    packs = [[] for i in range(number_of_options)]
    start = 0
    for i in range(len(Dm)):
        for j in range(start,start+int(Dm[i])):
            packs[i].append({"unit":j,"threshold":Dh[j]})
        start += int(Dm[i])

    return np.array(packs)

def vote_counter(Dx,assigned):
    votes = []
    for i in range(len(assigned)):
        count = 0
        for j in assigned[i]:
            if j["threshold"] < Dx[i]:
                count += 1
        votes.append(count)
    return votes

# Contants____________________________________________

population_size = 20000

number_of_options = 10

q = 5

mux = 100.0
sigmax = 5.0 

muh = 100.0
sigmah = 4.0

mum = 2000.0
sigmam = 0.0

Dx = np.random.normal(mux,sigmax,number_of_options)
Dh = np.random.normal(muh,sigmah,population_size)
Dm = dm(population_size,number_of_options)

assigned = units_assign_to_opt(Dh,Dm,number_of_options)

results = vote_counter(Dx,assigned)

option_votes = []
for i in range(number_of_options):
    option_votes.append({"xi":Dx[i],"votes":results[i]})


# option_votes = sorted(option_votes, key = lambda i: i['xi']) 


# Classes_____________________________________________



# Main program________________________________________

if __name__ == "__main__":
    # sum = 0
    # for m in Dm:
    #     sum += int(m)
    #     print(m)
    # print(sum)
    # print(assigned)
    # print(Dx)
    # print(assigned.shape)
    # print(option_votes)
    print(max(option_votes, key = lambda i: i['xi']))
    print(option_votes.index(max(option_votes, key = lambda i: i['xi'])))
    plt.scatter(Dx,results)
    plt.show()
    


# %%
