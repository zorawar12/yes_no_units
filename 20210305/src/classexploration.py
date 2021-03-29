# Nature of collective-decision making by simple 
# votes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

#%%

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest as pt

#%%

class Decision_making:
    def __init__(self,number_of_options,err_type,mu_assessment_err,sigma_assessment_err):
        self.err_type = err_type
        self.number_of_options = number_of_options
        self.quorum = None
        self.mu_assessment_err = mu_assessment_err
        self.sigma_assessment_err = sigma_assessment_err
        self.votes = None
        self.no_votes = None
        self.no_proportion = None
        self.yes_stats = []
        self.max_ratio_pvalue = None
        self.y_ratios = []

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

    def for_against_vote_counter(self,assigned_units,Dx,pc):
        """
        Each unit provides its decision and votes are counted for each options
        """
        votes = [0 for i in range(self.number_of_options)]
        no_votes = [0 for i in range(self.number_of_options)]
        for i in range(self.number_of_options):
            assesment_error = np.round(np.random.normal(self.mu_assessment_err,self.sigma_assessment_err,len(assigned_units[i])),decimals= self.err_type)
            for j in range(len(assigned_units[i])):
                if (assigned_units[i][j] < (Dx[i] + assesment_error[j])):
                    votes[i] += 1
                else:
                    no_votes[i] += 1

        self.votes = votes
        self.no_votes = no_votes
        self.no_proportion = [no_votes[i]/pc[i] for i in range(self.number_of_options)]
        self.hypothesis_testing(pc)
        self.max_ratio_pvalue = self.hypothesis_testing_top_two(pc)

    def hypothesis_testing(self,pc):
        for i in range(self.number_of_options-1):
            self.yes_stats.append([])
            for j in range(i+1,self.number_of_options):
                self.yes_stats[i].append(pt([self.votes[i],self.votes[j]],[pc[i],pc[j]]))

    def hypothesis_testing_top_two(self,pc):
        for j in range(self.number_of_options):
            self.y_ratios.append(self.votes[j]/pc[j])
        max_1 = [max(self.y_ratios),self.y_ratios.index(max(self.y_ratios))]
        self.y_ratios[max_1[1]] = 0
        max_2 = [max(self.y_ratios),self.y_ratios.index(max(self.y_ratios))]
        self.y_ratios[max_1[1]] = max_1[0]
        pvalue = pt([self.votes[max_1[1]],self.votes[max_2[1]]],[pc[max_1[1]],pc[max_2[1]]])

        return pvalue

    def best_among_bests_no(self,ref_highest_quality):
        """
        Returns success/failure of decision making when there are multiple correct decisions as per the units
        """
        if self.no_proportion.index(min(self.no_proportion)) ==  ref_highest_quality:
            return 1
        else:
            return 0

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
        best_list = np.array(np.where(self.Dx == max(self.Dx)))[0]
        opt_choosen = np.random.randint(0,len(best_list))
        self.ref_highest_quality = best_list[opt_choosen]

    def dx(self):
        """
        Provides distribution of quality stimulus for each option upto specified decimal places
        """
        self.Dx = np.round(np.random.normal(self.mu_x,self.sigma_x,self.number_of_options),decimals=self.x_type)


class Qranking:
    def __init__(self,number_of_options):
        self.n = number_of_options
        self.ref_rank = np.zeros((self.n,self.n))
        self.exp_rank = np.zeros((self.n,self.n))
        self.exp_rank_w_n = np.zeros((self.n,self.n))

    def ref_ranking(self,oq,y_ratios,no_votes):
        for i in range(len(oq)):
            for j in range(i+1,self.n):
                if oq[i]>oq[j]:
                    self.ref_rank[i,j] = 1
                if y_ratios[i]>y_ratios[j] and no_votes[i]<no_votes[j]:
                    self.exp_rank[i,j] = 1
                elif y_ratios[i]<y_ratios[j] and no_votes[i]>no_votes[j]:
                    self.exp_rank[i,j] = 0
                elif y_ratios[i]<y_ratios[j] and no_votes[i]<no_votes[j]:
                    self.exp_rank[i,j] = 0.5
                elif y_ratios[i]>y_ratios[j] and no_votes[i]>no_votes[j]:
                    self.exp_rank[i,j] = 0.5

    def ref_ranking_w_n(self,oq,y_ratios,no_votes):
        for i in range(len(oq)):
            for j in range(i+1,self.n):
                if y_ratios[i]>y_ratios[j]:
                    self.exp_rank_w_n[i,j] = 1
                elif y_ratios[i]<y_ratios[j]:
                    self.exp_rank_w_n[i,j] = 0
                else:
                    self.exp_rank_w_n[i,j] = 0.5

    def incorrectness_cost(self,exp_rank):
        measure_of_incorrectness = 0
        for i in range(self.n):
            for j in range(i+1,self.n):
                measure_of_incorrectness += abs(exp_rank[i,j]-self.ref_rank[i,j])
        measure_of_incorrectness = 2*measure_of_incorrectness/(self.n*(self.n - 1))
        return measure_of_incorrectness           #   Higher measure of incorrectness more bad is the ranking by units votes
# %%
