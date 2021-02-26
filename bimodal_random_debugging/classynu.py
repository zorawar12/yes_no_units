# Nature of collective-decision making by simple
# votes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

#%%

import numpy as np
import matplotlib.pyplot as plt
# from numba import jit,njit,vectorize

#%%

class Decision_making:
    def __init__(self,number_of_options,err_type,mu_assessment_err,\
        sigma_assessment_err):
        self.err_type = err_type
        self.number_of_options = number_of_options
        self.quorum = None
        self.mu_assessment_err = mu_assessment_err
        self.sigma_assessment_err = sigma_assessment_err
        self.votes = None

    def vote_counter(self,assigned_units,Dx):
        """
        Each unit provides its decision and votes are counted for each options
        """
        votes = [0 for i in range(self.number_of_options)]
        for i in range(self.number_of_options):
            assesment_error = np.round(np.random.normal(self.mu_assessment_err,self.sigma_assessment_err,len(assigned_units[i])),decimals= self.err_type)
            for j in range(len(assigned_units[i])):
                if (assigned_units[i][j] < (Dx[i] + assesment_error[j])):
                    votes[i] += 1
        self.votes = votes

    def best_among_bests(self,ref_highest_quality):
        """
        Returns success/failure of decision making when there are multiple correct decisions as per the units
        """
        available_opt = np.array(np.where(np.array(self.votes) == max(self.votes)))[0]
        opt_choosen = np.random.randint(0,len(available_opt))
        if available_opt[opt_choosen] ==  ref_highest_quality:
            return 1
        else:
            return 0

    def quorum_voting(self,assigned_units,Dx,ref_highest_quality):
        """
        success(1) or failure(0) ,quorum_reached(success(1) or failure(0)),majority decision (one_correct(success(1) or failure(0)),multi_correct(success(1) or failure(0)))
        """
        units_used = [0 for i in range(self.number_of_options)]
        for i in range(self.number_of_options):
            assesment_error = np.round(np.random.normal(self.mu_assessment_err,self.sigma_assessment_err,len(assigned_units[i])),decimals= self.err_type)
            count = 0
            while count<self.quorum:
                if units_used[i] == len(assigned_units[i]):
                    break
                if (assigned_units[i][units_used[i]] < (Dx[i] + assesment_error[units_used[i]])):
                    count += 1
                units_used[i] += 1
        units_used = np.array(units_used)
        loc = np.array(np.where(units_used == min(units_used)))[0]
        opt_choosen = np.random.randint(0,len(loc))

        flag = 0
        for i in range(self.number_of_options):
            if units_used[i] == len(assigned_units[i]):
                flag += 1
        if flag==self.number_of_options:
            quorum_reached = 0
            result = 0
            return result,quorum_reached

        if loc[opt_choosen] ==  ref_highest_quality:
            result = 1
            quorum_reached = 1
            return result,quorum_reached
        else:
            quorum_reached = 1
            result = 0
            return result,quorum_reached

#%%

class qualityControl:
    def __init__(self,number_of_options,x_type):
        self.mu_x_1 = None
        self.sigma_x_1 = None
        self.mu_x_2 = None
        self.sigma_x_2 = None
        self.Dx = None
        self.x_type = x_type
        self.number_of_options = number_of_options
        self.ref_highest_quality = None

    def ref_highest_qual(self):
        """
        Provides known highest quality option
        """
        best_list = np.array(np.where(self.Dx == max(self.Dx)))[0]
        opt_choosen = np.random.randint(0,len(best_list))
        self.ref_highest_quality = best_list[opt_choosen]

    def dx(self):
        """
        Provides distribution of quality stimulus for each option upto specified decimal places
        """
        self.Dx = []
        peak_choice = np.random.randint(0,2,self.number_of_options)
        # peak_choice = np.array([1 for i in range(int(self.number_of_options/2))])
        # peak_choice = np.append(peak_choice,np.array([0 for i in range(self.number_of_options-len(peak_choice))]))
        for i in peak_choice:
            if i==0:
                self.Dx.append(np.round(np.random.normal(self.mu_x_1,self.sigma_x_1),decimals=self.x_type))
            else:
                self.Dx.append(np.round(np.random.normal(self.mu_x_2,self.sigma_x_2),decimals=self.x_type))

