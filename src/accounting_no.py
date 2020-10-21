# Nature of collective-decision making by simple yes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

import numpy as np
import matplotlib.pyplot as plt
if __name__ != "__main__":    
    import src.classynu as yn
else:
    import classynu as yn
from multiprocessing import Pool
from operator import itemgetter 

#%%
# Accounting for no units along with yes units

number_of_options = 10                      #   Number of options to choose best one from
mu_x = 0.0                                  #   Mean of distribution from which quality stimulus are sampled randomly
sigma_x = 1.0                               #   Standard deviation of distribution from which quality stimulus are sampled randomly
mu_h = 0                                    #   Mean of distribution from which units threshold are sampled randomly
sigma_h = 1.0                               #   Standard deviation of distribution from which units threshold are sampled randomly
mu_m = 100                                  #   Mean of distribution from which number of units to be assigned to an option are sampled randomly
sigma_m = 30                                 #   Standard deviation of distribution from which number of units to be assigned to an option are sampled randomly
mu_assessment_err = 0.0                     #   Mean of distribution from which units quality assessment error are sampled randomly
sigma_assessment_err = 0.0                  #   Standard deviation of distribution from which units quality assessment error are sampled randomly
x_type = 3                                  #   Number of decimal places of quality stimulus
h_type = 3                                  #   Number of decimal places of units threshold
err_type = 0                                #   Number of decimal places of quality assessment error

#%%
if __name__ == "__main__":
    success_rate_sig_h_number_options = 1
#%%

def units(mu_m,sigma_m,number_of_options):
    """
    Arguments:
    mu_m -(int) mean of distribution from where to choose number of units to be assigned
    sigma_m - (int) standard deviation of distribution from where to choose number of units to be assigned
    Returns:
    Number of units to be assigned to an option choosen from N(mu_m,sigma_m) (array[1xn])
    """
    return np.round(np.random.normal(mu_m,sigma_m,number_of_options),decimals=0)

def threshold(m_units,h_type,mu_h,sigma_h):
    """
    Creates threshold distribution
    Arguments:
    m_units - (int from [1xn])number of assigned units to an option
    h_type - (int) number of decimal places of thresholds
    mu_h - (float) mean of distribution to select threshold for each unit from
    sigma_h - (float) standard deviation of distribution to select threshold for each unit from

    Returns: 
    Array[1xm_units] of thesholds for each assigned set of units to options
    """
    return np.round(np.random.normal(mu_h,sigma_h,abs(m_units)),decimals=h_type)

def quality(number_of_options,x_type,mu_x,sigma_x):
    """
    Creates quality stimulus
    Arguments:
    number_of_options - (int)number of option for which quality stimulus has to be assigned
    x_type - (int) number of decimal places of quality stimulus
    mu_x - (float) mean of distribution to select quality stimulus for each option from
    sigma_x - (float) standard deviation of distribution to select quality stimulus for each option from
    Returns: 
    Array[1 x number_of_options] of qualities for each option
    """
    QC = yn.qualityControl(number_of_options=number_of_options,x_type=x_type)
    QC.mu_x = mu_x
    QC.sigma_x = sigma_x
    QC.dx()
    QC.ref_highest_qual()
    return QC

def majority_decision_no(number_of_options,Dx,assigned_units,err_type,\
    mu_assessment_err,sigma_assessment_err,ref_highest_quality,\
    quorum = None):
    """
    Majority based decision

    Arguments:
    number_of_options - (int)number of options to choose best from
    Dx - ([1 x number_of_options]) array of quality stimulus for each options (options sorted from highest quality to lowest quality)
    assigned_units - ([[1 x m_units] for i in range(len(number_of_options)]) array of assigned units with thresholds to each options
    err_type - (int) number of decimal places of noise
    mu_assessment_err - (float) mean of distribution to choose noise from
    sigma_assessment_err - (float) standard deviation of distribution to choose noise from
    ref_highest_quality - highest quality option to measure success
    quorum - (int) quorum to be reached first in order to consider an option as best
    Returns:
    If quorum = None then success(1) or failure(0)
    else success(1) or failure(0) ,quorum_reached(success(1) or failure(0)),majority decision (one_correct(success(1) or failure(0)),multi_correct(success(1) or failure(0)))
    """
    DM = yn.Decision_making(number_of_options=number_of_options,err_type=err_type,\
    mu_assessment_err=mu_assessment_err,sigma_assessment_err=sigma_assessment_err)
    DM.quorum = quorum
    DM.vote_counter(assigned_units,Dx)
    DM.for_against_vote_counter(assigned_units,Dx)
    majority_dec = DM.best_among_bests_no(ref_highest_quality)
    if quorum == None:
        # plt.scatter(Dx,DM.votes)
        # plt.show()
        return majority_dec
    else:
        result,quorum_reached = DM.quorum_voting(assigned_units,Dx,ref_highest_quality)
        return result,quorum_reached,majority_dec

def multi_run_no(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h,\
    x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,err_type=err_type,mu_assessment_err= mu_assessment_err,\
    sigma_assessment_err=sigma_assessment_err,quorum= None):

    pc = np.array(units(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m)).astype(int)

    units_distribution = []
    for i in pc:
        units_distribution.append(threshold(m_units = i ,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h))

    qc = quality(number_of_options=number_of_options,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x)

    dec = majority_decision_no(number_of_options=number_of_options,Dx = qc.Dx,assigned_units= units_distribution,\
        err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err,\
        ref_highest_quality=qc.ref_highest_quality,quorum=quorum)  

    return dec

def plt_show(data_len,array,var,plt_var,x_name,title,save_name):
    c = ["blue","green","red","purple","brown"]
    count = 0
    fig = plt.figure()
    data = [[] for i in range(len(data_len))]

    for i in array:
        data[data_len.index(i[var])].append(i)

    for i in data:
        plt.scatter(list(map(itemgetter(plt_var), i)),list(map(itemgetter("success_rate"), i)),c = c[count],s=0.3)    
        count += 1

    plt.xlabel(x_name)
    plt.ylabel('Rate_of_correct_choice')
    plt.legend(data_len,markerscale = 10, title = title)
    plt.savefig(save_name,format = "pdf")
    plt.show()

def parallel(func,a,b):
    inp = []
    for i in a:
        for j in b:
            inp.append((i,j))

    opt_var = []

    with Pool(8) as p:
        opt_var = p.starmap(func,inp)
    
    return opt_var

def graphic_plt(a,b,array,x_name,y_name,title,save_name):
    fig, ax = plt.subplots()
    z = np.array(list(map(itemgetter("success_rate"), array))).reshape(len(a),len(b))
    cs = ax.contourf(b,a,z)   
    cbar = fig.colorbar(cs)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend(title = title)
    plt.savefig(save_name,format = "pdf")
    plt.show()

def bar(quor,opt_v,save_name,correct):
    fig, ax = plt.subplots()
    ax.bar(quor,[1 for i in range(1,101,1)],width=1,color = "white",edgecolor='black')
    ax.bar(quor,list(map(itemgetter("success_rate"), opt_v)),width=1,color = "blue",edgecolor='black')
    ax.bar(quor,list(map(itemgetter("q_not_reached"), opt_v)),width=1,color = "orange",edgecolor='black',bottom = list(map(itemgetter("success_rate"), opt_v)))
    plt.plot(quorum,list(map(itemgetter(correct), opt_v)),color ="red")
    plt.xlabel('Quorum')
    plt.ylabel('Rate of choice')
    # plt.legend(title = leg)
    plt.savefig(save_name,format = "pdf")
    plt.show()
#%%

# Majority based Rate of correct choice as a function of sigma_h for varying number of options
if success_rate_sig_h_number_options==1:    
    sig_h = [0.0+i*0.01 for i in range(101)]
    opts = [2,10]#2*i for i in range(2,6,3)]

    def sighf(op,sigh):
        count = 0
        for k in range(1000):
            success = multi_run_no(sigma_h=sigh,number_of_options=op,err_type=0) 
            if success == 1:
                count += 1
        opt_va = {"opt":op,"sigma": sigh, "success_rate":count/1000}
        return opt_va

    opt_var = parallel(sighf,opts,sig_h)

    plt_show(data_len= opts,array= opt_var,var= "opt", plt_var="sigma",x_name='Sigma_h',\
        title="Number of options",save_name="Sigma_h_vs_Rate_of_correct_choice_sorted_no.pdf")

