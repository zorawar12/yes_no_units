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

    def vote_counter(self,assigned_units,Dx):
        """
        Each unit provides its decision and votes are counted for each options 
        """
        votes = []
        for i in range(len(assigned_units)):
            count = 0
            assesment_error = np.round(np.random.normal(self.mu_assessment_err,self.sigma_assessment_err,len(assigned_units[i])),decimals= self.err_type)
            for j in range(len(assigned_units[i])):
                if (assigned_units[i][j] < (Dx[i] + assesment_error[j])):
                    count += 1
            votes.append(count)
        self.votes = votes

    def one_correct(self,ref_highest_quality):
        """
        Returns success/failure of decision making when there is only one correct decision
        """
        if np.array(np.where(np.array(self.votes) == max(self.votes)))[0][0] == ref_highest_quality:
            return 1

    def multi_correct(self,ref_highest_quality):
        """
        Returns success/failure of decision making when there are multiple correct decisions as per the units
        """        
        available_opt = np.array(np.where(np.array(self.votes) == max(self.votes)))[0]
        opt_choosen = np.random.randint(0,len(available_opt))
        if available_opt[opt_choosen] ==  ref_highest_quality:
            return 1

    # def quorum_decision(self,assigned_units,Dx):
    #     """
    #     Each unit provides its decision and votes are counted for each options 
    #     """
    #     votes = []
    #     for i in range(len(assigned_units)):
    #         count = 0
    #         assesment_error = np.round(np.random.normal(self.mu_assessment_err,self.sigma_assessment_err,len(assigned_units[i])),decimals= self.err_type)
    #         for j in range(len(assigned_units[i])):
    #             if assigned_units[i][j]["threshold"]  < Dx[i] + assesment_error[j] :
    #                 count += 1
    #         votes.append(count)
    #         for i in range(self.number_of_options):
    #             if votes[i]>=self.quorum:
    #                 return i

    

#%%

class qualityControl:
    def __init__(self,number_of_options,x_type):
        self.mu_x = None
        self.sigma_x = None
        self.Dx = None
        self.x_type = x_type
        self.number_of_options = number_of_options
        self.ref_highest_quality = None

    def ref_highest_qual(self):
        """
        Provides known highest quality option
        """
        self.ref_highest_quality = np.array(np.where(self.Dx == max(self.Dx)))[0][0]

    def dx(self):
        """
        Provides distribution of quality stimulus for each option upto specified decimal places
        """        
        self.Dx = np.sort(np.round(np.random.normal(self.mu_x,self.sigma_x,self.number_of_options),decimals=self.x_type))


#%%

# class visualize:
#     def __init__(self):


