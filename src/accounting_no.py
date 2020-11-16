# Nature of collective-decision making by simple yes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

import numpy as np
import matplotlib.pyplot as plt
if __name__ != "__main__":    
    import src.classexploration as yn
else:
    import classexploration as yn
from multiprocessing import Pool
from operator import itemgetter 


# Accounting for no units along with yes units

number_of_options = 10                      #   Number of options to choose best one from
mu_x = 0.0                                  #   Mean of distribution from which quality stimulus are sampled randomly
sigma_x = 1.0                               #   Standard deviation of distribution from which quality stimulus are sampled randomly
mu_h = 0                                    #   Mean of distribution from which units threshold are sampled randomly
sigma_h = 1.0                               #   Standard deviation of distribution from which units threshold are sampled randomly
mu_m = 150                                  #   Mean of distribution from which number of units to be assigned to an option are sampled randomly
sigma_m = 10                                #   Standard deviation of distribution from which number of units to be assigned to an option are sampled randomly
mu_assessment_err = 0.0                     #   Mean of distribution from which units quality assessment error are sampled randomly
sigma_assessment_err = 0.0                  #   Standard deviation of distribution from which units quality assessment error are sampled randomly
x_type = 3                                  #   Number of decimal places of quality stimulus
h_type = 3                                  #   Number of decimal places of units threshold
err_type = 0                                #   Number of decimal places of quality assessment error
confidence = 0.02                           #   Confidence for distinguishing qualities

success_rate_mu_m_number_options = 1


def units(mu_m,sigma_m,number_of_options):
    """
    Arguments:
    mu_m -(int) mean of distribution from where to choose number of units to be assigned
    sigma_m - (int) standard deviation of distribution from where to choose number of units to be assigned
    Returns:
    Number of units to be assigned to an option choosen from N(mu_m,sigma_m) (array[1xn])
    """
    a = np.array(abs(np.round(np.random.normal(mu_m,sigma_m,number_of_options),decimals=0)))
    while a.any()==0.:
        a = np.array(abs(np.round(np.random.normal(mu_m,sigma_m,number_of_options),decimals=0)))
    # print(a)
    return a

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

def decision_make_check(number_of_options,Dx,assigned_units,err_type,mu_assessment_err,sigma_assessment_err,\
    ref_highest_quality,pc = None,quorum = None):
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
    DM.for_against_vote_counter(assigned_units,Dx,pc)
    majority_dec = DM.best_among_bests_no(ref_highest_quality)
    qrincorrectness = yn.Qranking(number_of_options)
    qrincorrectness.ref_rank(Dx,DM.y_ratios,DM.no_votes)
    incorrectness = qrincorrectness.incorrectness_cost()

    if quorum == None:
        # plt.scatter(Dx,DM.votes)
        # plt.show()
        return majority_dec,DM.yes_stats,DM.max_ratio_pvalue,incorrectness
    else:
        result,quorum_reached = DM.quorum_voting(assigned_units,Dx,ref_highest_quality)
        return result,quorum_reached,majority_dec

def main_process_flow(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h,\
    x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,err_type=err_type,mu_assessment_err= mu_assessment_err,\
    sigma_assessment_err=sigma_assessment_err,quorum= None):

    pc = np.array(units(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m)).astype(int)

    units_distribution = []
    for i in pc:
        units_distribution.append(threshold(m_units = i ,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h))

    qc = quality(number_of_options=number_of_options,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x)

    dec = decision_make_check(number_of_options=number_of_options,Dx = qc.Dx,assigned_units= units_distribution,\
        err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err,\
        ref_highest_quality=qc.ref_highest_quality,quorum=quorum,pc = pc)
    
    return dec


def plt_show(data_len,array,var,plt_var,x_name,title,save_name,y_var):
    c = ["blue","green","red","purple","brown"]
    count = 0
    fig = plt.figure()
    data = [[] for i in range(len(data_len))]

    for i in array:
        data[data_len.index(i[var])].append(i)

    for i in data:
        plt.scatter(list(map(itemgetter(plt_var), i)),list(map(itemgetter(y_var), i)),c = c[count],s=10)    
        count += 1
    plt.ylim(top = 1,bottom = -0.2)
    plt.xlabel(x_name)
    plt.ylabel('P-Values')
    plt.legend(data_len,markerscale = 5, title = title)
    plt.savefig(save_name,format = "pdf")
    plt.show()

def parallel(func,a,b):
    inp = []
    for i in a:
        for j in b:
            inp.append((i,j))

    opt_var = []

    for i in inp:
        opt_var.append(func(i[0],i[1]))

    # with Pool(8) as p:
    #     opt_var = p.starmap(func,inp)
    
    return opt_var

if __name__ != "__main__":
    # Majority based Rate of correct choice as a function of sigma_h for varying number of options
    sig_h = [0.0+i*0.01 for i in range(101)]
    opts = [5,10]#2*i for i in range(2,6,3)]

    def sighf(op,sigh):
        count = 0
        for k in range(1000):
            success,yes_test = main_process_flow(sigma_h=sigh,number_of_options=op,err_type=0)
            print(yes_test)
            if success == 1:
                count += 1
        return {"opt":op,"sigma": sigh, "success_rate":count/1000}

    output = parallel(sighf,opts,sig_h)

    plt_show(data_len= opts,array= output,var= "opt", plt_var="sigma",x_name='Sigma_h',\
        title="Number of options",save_name="Sigma_h_vs_Rate_of_correct_choice_sorted_no.pdf")

if success_rate_mu_m_number_options==1:
    mu_m = [i for i in range(50,10000,200)]
    number_of_options = [5,10]

    def mumf(nop,mum):
        count = 0
        sum_pval = 0
        avg_incrtness = 0
        for k in range(2000):
            success ,yes_test,max_rat_pval,incrt = main_process_flow(mu_m=mum,number_of_options=nop,err_type=0)
            # print(max_rat_pval)
            # if max_rat_pval[1] != 'nan' or 'inf':
            sum_pval += max_rat_pval[1]
            avg_incrtness += incrt
            # max_p=[[max(i[0]),i.index(max(i[0])),yes_test.index(i)] for i in yes_test]
            # print(max_p)
            # max_p_overall = max(max_p)
            if success == 1:
                count += 1
        avg_pval = sum_pval/2000
        avg_incrtness = avg_incrtness/2000
        return {"nop":nop,"mum": mum, "success_rate":count/2000,'avg_pval':avg_pval,'avg_incrt':avg_incrtness}

    opt_var = parallel(mumf,number_of_options,mu_m)
    # print(opt_var)

    plt_show(data_len= number_of_options,array= opt_var,var= "nop", plt_var="mum",x_name='mean_number_of_units(variance = 10)',\
        title="Number_of_options",save_name="number_of_units_vs_pvalue.pdf",y_var="avg_pval")

    plt_show(data_len= number_of_options,array= opt_var,var= "nop", plt_var="mum",x_name='mean_number_of_units(variance = 10)',\
        title="Number_of_options",save_name="number_of_units_vs_pvalue.pdf",y_var="avg_incrt")
