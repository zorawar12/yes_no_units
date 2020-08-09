# Nature of collective-decision making by simple 
# yes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin12a@iiserb.ac.in

#%%

import numpy as np
import matplotlib.pyplot as plt
import classynu as yn



def majority_decision():
    """
    Majority based decision without any assesment error
    """
    ynu = yn.Decision_making(population_size = 2000,number_of_options = 10)
    ynu.mu_x = 0.0
    ynu.sigma_x = 1.0
    ynu.mu_h = 0.0
    ynu.sigma_h = 1.0
    ynu.mu_m = 200.0
    ynu.sigma_m = 0.0
    ynu.dm()
    ynu.dh()
    ynu.dx()
    ynu.ref_highest_qual()
    ynu.units_assign_to_opt()
    ynu.mu_assessment_err = 0.0
    ynu.sigma_assessment_err = 0.0
    ynu.vote_counter()
    ynu.vote_associator()

    if np.where(ynu.vote_for_opt == max(ynu.vote_for_opt, key = lambda i: i['votes']))[0][0] == ynu.ref_highest_quality:
        print("Success")
    else:
        print("Failure")
    plt.scatter(ynu.Dx,ynu.votes)
    plt.show()


majority_decision()




# num_of_units = [j for j in range(100,20000,500)]
# num_of_opt = [j for j in range(2,20,1)]


# for j in num_of_units:
#     for i in num_of_opt:
#         ynu = yn.Decision_making(j,i)



# %%
