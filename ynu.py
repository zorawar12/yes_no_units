# Nature of collective-decision making by simple 
# yes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin12a@iiserb.ac.in

#%%

import numpy as np
import matplotlib.pyplot as plt
import classynu as yn

#%%

def majority_decision(population_size = 2000,number_of_options = 10,mu_x = 0.0,\
    sigma_x = 1.0,mu_h = 0.0,sigma_h = 1.0,mu_m = 200,sigma_m = 0,mu_assessment_err = 0.0,\
    sigma_assessment_err = 0.0,x_type = 3,h_type = 3, m_type = 3,err_type = 3,one_correct_opt = 1):
    """
    Majority based decision
    """
    ynu = yn.Decision_making(population_size,number_of_options,x_type,h_type, m_type,err_type)
    ynu.mu_x = mu_x
    ynu.sigma_x = sigma_x
    ynu.mu_h = mu_h
    ynu.sigma_h = sigma_h
    ynu.mu_m = mu_m
    ynu.sigma_m = sigma_m
    ynu.dm()
    ynu.dh()
    ynu.dx()
    ynu.ref_highest_qual()
    ynu.units_assign_to_opt()
    ynu.mu_assessment_err = mu_assessment_err
    ynu.sigma_assessment_err = sigma_assessment_err
    ynu.vote_counter()
    ynu.vote_associator()

    if one_correct_opt == 1:
        if ynu.one_correct() == 1:
            print("Success")
        else:
            print("Failure")
        
    else:
        if ynu.multi_correct() == 1:
            print("Success")
        else:
            print("Failure")
    
    plt.scatter(ynu.Dx,ynu.votes)
    plt.show()

#%%

majority_decision()                                   # Without assesment error

#%%

majority_decision(sigma_assessment_err = 5)           # With assessment error

#%%

majority_decision(x_type = 0, m_type = 0,err_type = 0,one_correct_opt = 0)        # Random choice when more than one option's correct

#%%

# num_of_units = [j for j in range(100,20000,500)]
# num_of_opt = [j for j in range(2,20,1)]


# for j in num_of_units:
#     for i in num_of_opt:
#         ynu = yn.Decision_making(j,i)



# %%
