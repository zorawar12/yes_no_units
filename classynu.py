# Nature of collective-decision making by simple 
# yes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin12a@iiserb.ac.in

#%%

import numpy as np
import matplotlib.pyplot as plt

#%%

class Decision_making:
    def __init__(self,number_of_options,err_type,mu_assessment_err,sigma_assessment_err):
        self.err_type = err_type
        self.number_of_options = number_of_options
        self.quorum = None
        self.mu_assessment_err = mu_assessment_err
        self.sigma_assessment_err = sigma_assessment_err
        self.votes = None
        self.vote_for_opt = None

    def vote_counter(self,assigned_units,Dx):
        """
        Each unit provides its decision and votes are counted for each options 
        """
        votes = []
        for i in range(len(assigned_units)):
            count = 0
            assesment_error = np.round(np.random.normal(self.mu_assessment_err,self.sigma_assessment_err,len(assigned_units[i])),decimals= self.err_type)
            for j in range(len(assigned_units[i])):
                if assigned_units[i][j]["threshold"]  < Dx[i] + assesment_error[j] :
                    count += 1
            votes.append(count)
        self.votes = votes

    def vote_associator(self,Dx):
        """
        Associates option with number of votes it received
        """
        option_votes = []
        for i in range(self.number_of_options):
            option_votes.append({"xi":Dx[i],"votes":self.votes[i]})
        self.vote_for_opt = np.array(option_votes)

    def one_correct(self,ref_highest_quality):
        """
        Returns success/failure of decision making when there is only one correct decision
        """
        if np.where(self.vote_for_opt == max(self.vote_for_opt, key = lambda i: i['votes']))[0][0] == ref_highest_quality:
            return 1

    def multi_correct(self,ref_highest_quality):
        """
        Returns success/failure of decision making when there are multiple correct decisions as per the units
        """        
        available_opt = np.where(self.vote_for_opt == max(self.vote_for_opt, key = lambda i: i['votes']))[0]
        opt_choosen = np.random.randint(0,len(available_opt))
        if available_opt[opt_choosen] ==  ref_highest_quality:
            return 1


#%%

class populationControl:
    def __init__(self,population_size,number_of_options):
        self.population_size = population_size
        self.number_of_options = number_of_options
        self.mu_m = None
        self.sigma_m = None
        self.Dm = None

    def dm(self):
        """
        Provides distribution of m such that total number of choosen 
        units is less than or equal to the population size
        """
        sum = self.population_size + 1
        while sum > self.population_size:
            Dm = np.round(np.random.normal(self.mu_m,self.sigma_m,self.number_of_options),decimals=0)
            sum = np.sum(Dm)
        self.Dm = Dm.astype("int32")


#%%

class thresholdControl:
    def __init__(self,population_size,h_type):
        self.Dh = None
        self.mu_h = None
        self.sigma_h = None
        self.population_size = population_size
        self.h_type = h_type

    def dh(self):
        """
        Provides distribution of thresholds for each unit upto specified decimal places
        """
        self.Dh = np.round(np.random.normal(self.mu_h,self.sigma_h,self.population_size),decimals=self.h_type) 


#%%

class qualityControl:
    def __init__(self,number_of_options,x_type):
        self.mu_x = None
        self.sigma_x = None
        self.Dx = None
        self.x_type = x_type
        self.number_of_options = number_of_options
        self.ref_highest_quality = None
        self.assigned_units = None

    def ref_highest_qual(self):
        """
        Provides known highest quality option
        """
        self.ref_highest_quality = np.where(self.Dx == max(self.Dx))[0][0]

    def dx(self):
        """
        Provides distribution of quality stimulus for each option upto specified decimal places
        """        
        self.Dx = np.sort(np.round(np.random.normal(self.mu_x,self.sigma_x,self.number_of_options),decimals=self.x_type))

    def assign_units_to_opts(self,Dm,Dh):
        """
        Assigns units to options along with their thresholds 
        """
        packs = [[] for i in range(self.number_of_options)]
        start = 0
        for i in range(len(Dm)):
            for j in range(start,start+Dm[i]):
                packs[i].append({"unit":j,"threshold":Dh[j]})
            start += Dm[i]

        self.assigned_units = np.array(packs)


#%%


#%%

# class visualize:
#     def __init__(self):


