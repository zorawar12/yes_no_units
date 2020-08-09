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
            assesment_error = np.random.normal(0,1.0,1)
            if j["threshold"]  < Dx[i] + assesment_error[0] :
                count += 1
        votes.append(count)
    return votes



# Classes_____________________________________________



# Main program________________________________________

if __name__ == "__main__":
    success = 0
    population_size = 2000

    number_of_options = 2

    q = 5

    mux = 0.0
    sigmax = 0.1

    muh = -3.0
    sigmah = 0.4

    mum = 200.0
    sigmam = 0.0

    for l in range(1000):    
        # Contants____________________________________________

        Dx = np.random.normal(mux,sigmax,number_of_options)
        ref_highest_quality = np.where(Dx == max(Dx))[0][0]
        # print(ref_highest_quality)
        Dh = np.random.normal(muh,sigmah,population_size)
        Dm = dm(population_size,number_of_options)

        assigned = units_assign_to_opt(Dh,Dm,number_of_options)

        results = vote_counter(Dx,assigned)

        option_votes = []
        for i in range(number_of_options):
            option_votes.append({"xi":Dx[i],"votes":results[i]})


        # option_votes = sorted(option_votes, key = lambda i: i['xi']) 


        # sum = 0
        # for m in Dm:
        #     sum += int(m)
        #     print(m)
        # print(sum)
        # print(assigned)
        # print(Dx)
        # print(assigned.shape)
        # print(option_votes)
        # print(max(option_votes, key = lambda i: i['xi']))
        # print(option_votes.index(max(option_votes, key = lambda i: i['xi'])))

        if option_votes.index(max(option_votes, key = lambda i: i['votes'])) == ref_highest_quality:
            # print("Success")
            success += 1
        # else:
            # print("Failure")
        # plt.scatter(Dx,results)
        # plt.show()
    print(success/(l+1))
    


# %%


# %%
