# Nature of collective-decision making by simple 
# yes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin12a@iiserb.ac.in

#%%

import numpy as np
import matplotlib.pyplot as plt

#%%

class Decision_making:
    def __init__(self,population_size,number_of_options,x_type,h_type, m_type ,err_type):
        self.x_type = x_type
        self.h_type = h_type
        self.m_type = m_type
        self.err_type = err_type
        self.population_size = population_size
        self.number_of_options = number_of_options
        self.quorum = None
        self.mu_x = None
        self.sigma_x = None
        self.mu_h = None
        self.sigma_h = None
        self.mu_m = None
        self.sigma_m = None
        self.Dm = None
        self.Dh = None
        self.Dx = None
        self.assigned_units = None
        self.mu_assessment_err = None
        self.sigma_assessment_err = None
        self.votes = None
        self.ref_highest_quality = None
        self.vote_for_opt = None

    def ref_highest_qual(self):
        self.ref_highest_quality = np.where(self.Dx == max(self.Dx))[0][0]

    def dh(self):
        self.Dh = np.round(np.random.normal(self.mu_h,self.sigma_h,self.population_size),decimals=self.h_type)

    def dx(self):
        self.Dx = np.round(np.random.normal(self.mu_x,self.sigma_x,self.number_of_options),decimals=self.x_type)

    def dm(self):
        """
        Provides distribution of m such that total number of choosen 
        units is less than or equal to the population size
        """
        sum = self.population_size + 1
        while sum > self.population_size:
            Dm = np.round(np.random.normal(self.mu_m,self.sigma_m,self.number_of_options),decimals=self.m_type)
            sum = 0
            for m in Dm:
                sum += int(m)
        self.Dm = Dm

    def units_assign_to_opt(self):
        """
        Assigns units to options along with their thresholds 
        """
        packs = [[] for i in range(self.number_of_options)]
        start = 0
        for i in range(len(self.Dm)):
            for j in range(start,start+int(self.Dm[i])):
                packs[i].append({"unit":j,"threshold":self.Dh[j]})
            start += int(self.Dm[i])

        self.assigned_units = np.array(packs)

    def vote_counter(self):
        """
        Each unit provides its decision and votes are counted for each options 
        """
        votes = []
        for i in range(len(self.assigned_units)):
            count = 0
            for j in self.assigned_units[i]:
                assesment_error = np.round(np.random.normal(self.mu_assessment_err,self.sigma_assessment_err,1),decimals= self.err_type)[0]
                if j["threshold"]  < self.Dx[i] + assesment_error :
                    count += 1
            votes.append(count)
        self.votes = votes

    def vote_associator(self):
        option_votes = []
        for i in range(self.number_of_options):
            option_votes.append({"xi":self.Dx[i],"votes":self.votes[i]})
        self.vote_for_opt = np.array(option_votes)

    def one_correct(self):
        if np.where(self.vote_for_opt == max(self.vote_for_opt, key = lambda i: i['votes']))[0][0] == self.ref_highest_quality:
            return 1

    def multi_correct(self):
        available_opt = np.where(self.vote_for_opt == max(self.vote_for_opt, key = lambda i: i['votes']))[0]
        opt_choosen = np.random.randint(0,len(available_opt))
        if available_opt[opt_choosen] ==  self.ref_highest_quality:
            return 1

#%%

# class visualize:
#     def __init__(self):


