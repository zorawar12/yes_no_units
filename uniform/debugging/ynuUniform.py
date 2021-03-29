# Nature of collective-decision making by simple yes/no decision units (Hasegawa paper).

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import classUniform as yn
from multiprocessing import Pool
from operator import itemgetter
import pandas as pd
import os


number_of_options = 10                      #   Number of options to choose best one from
low_x = -np.sqrt(3)                         #   Lower bound of distribution from which quality stimulus are sampled randomly
high_x = np.sqrt(3)                         #   Upper bound of distribution from which quality stimulus are sampled randomly
low_h = -np.sqrt(3)                         #   Lower bound of distribution from which units threshold are sampled randoml
high_h = np.sqrt(3)                         #   Upper bound of distribution from which units threshold are sampled randomly
low_m = 100                                 #   Lower bound of distribution from which number of units to be assigned to an option are sampled randomly
high_m = 100                                #   Upper bound of distribution from which number of units to be assigned to an option are sampled randomly
low_assessment_err = -1.0                   #   Lower bound of distribution from which units quality assessment error are sampled randomly
high_assessment_err = 1.0                   #   Upper bound of distribution from which units quality assessment error are sampled randomly
x_type = 3                                  #   Number of decimal places of quality stimulus
h_type = 3                                  #   Number of decimal places of units threshold
err_type = 0                                #   Number of decimal places of quality assessment error
path = os.getcwd() + "/results/"

Without_assesment_error_Majority_based_decision = 0
With_assesment_error_Majority_based_decision = 0
random_choice_many_best_option = 0
sigma_h_vs_RCD_vs_nop = 0
mu_h_vs_RCD_vs_nop = 0
nop_vs_RCD_vs_mu_h = 0
mu_m_vs_RCD_vs_nop = 0
sigma_m_vs_RCD_vs_nop = 0
sigma_h_vs_mu_h_vs_RCD = 0
mu_x_vs_mu_h_vs_RCD = 1
sigma_x_vs_sigma_h_vs_RCD = 0
quorum_vs_RC_vs_sigma_m = 0


def units(low_m,high_m,number_of_options):
    return np.round(np.random.uniform(low=low_m,high=high_m,size=number_of_options),decimals=0)

def threshold(m_units,h_type,low_h,high_h):
    return np.round(np.random.uniform(low=low_h,high=high_h,size=abs(m_units)),decimals=h_type)

def quality(number_of_options,x_type,low_x,high_x):
    QC = yn.qualityControl(number_of_options=number_of_options,x_type=x_type)
    QC.low_x = low_x
    QC.high_x = high_x
    QC.dx()
    QC.ref_highest_qual()
    return QC

def majority_decision(number_of_options,Dx,assigned_units,err_type,\
    low_assessment_err,high_assessment_err,ref_highest_quality,\
    quorum = None):

    DM = yn.Decision_making(number_of_options=number_of_options,err_type=err_type,\
    low_assessment_err=low_assessment_err,high_assessment_err=high_assessment_err)
    DM.quorum = quorum
    DM.vote_counter(assigned_units,Dx)
    majority_dec = DM.best_among_bests(ref_highest_quality)
    if quorum == None:
        #plt.scatter(Dx,DM.votes)
        #plt.show()

        return majority_dec

    else:
        result,quorum_reached = DM.quorum_voting(assigned_units,Dx,ref_highest_quality)
        return result,quorum_reached,majority_dec

def one_run(number_of_options=number_of_options,low_m=low_m,high_m=high_m,h_type=h_type,low_h=low_h,high_h=high_h,\
    x_type=x_type,low_x=low_x,high_x=high_x,err_type=err_type,low_assessment_err=low_assessment_err,high_assessment_err=high_assessment_err):

    pc = np.array(units(number_of_options=number_of_options,low_m=low_m,high_m=high_m)).astype(int)

    units_distribution = []
    for i in pc:
        units_distribution.append(threshold(m_units = i ,h_type=h_type,low_h=low_h,high_h=high_h))

    qc = quality(number_of_options=number_of_options,x_type=x_type,low_x=low_x,high_x=high_x)

    dec = majority_decision(number_of_options=number_of_options,Dx = qc.Dx,assigned_units= units_distribution,\
        err_type=err_type,low_assessment_err=low_assessment_err,high_assessment_err=high_assessment_err,\
        ref_highest_quality=qc.ref_highest_quality)

    if dec == 1:
        print("success")

    else:
        print("failed")

def multi_run(number_of_options=number_of_options,low_m=low_m,high_m=high_m,h_type=h_type,low_h=low_h,high_h=high_h,\
    x_type=x_type,low_x=low_x,high_x=high_x,err_type=err_type,low_assessment_err=low_assessment_err,high_assessment_err=high_assessment_err,quorum= None):

    pc = np.array(units(number_of_options=number_of_options,low_m=low_m,high_m=high_m)).astype(int)

    units_distribution = []
    for i in pc:
        units_distribution.append(threshold(m_units = i ,h_type=h_type,low_h=low_h,high_h=high_h))

    qc = quality(number_of_options=number_of_options,x_type=x_type,low_x=low_x,high_x=high_x)

    dec = majority_decision(number_of_options=number_of_options,Dx = qc.Dx,assigned_units= units_distribution,\
        err_type=err_type,low_assessment_err=low_assessment_err,high_assessment_err=high_assessment_err,\
        ref_highest_quality=qc.ref_highest_quality,quorum=quorum)

    return dec


def parallel(func,a,b):
    inp = []
    for i in a:
        for j in b:
            inp.append((i,j))

    opt_var = []

    with Pool(20) as p:
        opt_var = p.starmap(func,inp)

    return opt_var


def graphicPlot(a,b,array,x_name,y_name,title,save_name,bar_label):
    fig, ax = plt.subplots()
    z = np.array(list(map(itemgetter("success_rate"), array))).reshape(len(a),len(b))
    cs = ax.contourf(b,a,z)
    cbar = fig.colorbar(cs)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend(title = title)
    cbar.set_label(bar_label)
    plt.savefig(save_name,format = "pdf")
    plt.show()


def csv(data,file):
    f = pd.DataFrame(data=data)
    f.to_csv(file)



# Majority based Rate of correct choice as a function of mu_x for varying mu_h
if mu_x_vs_mu_h_vs_RCD==1:
    mu_x = [np.round(i*0.1,decimals=2) for i in range(51)]
    mu_h = [np.round(i*0.1,decimals=2) for i in range(51)]

    def lowhxf(muh,mux):
        count = 0
        for k in range(500):
            success = multi_run(low_h=muh+low_h,low_x=mux+low_x,high_h=muh+high_h,high_x=mux+high_x,err_type=0)
            if success == 1:
                count += 1
        mu_va = {"mux":mux,"muh": muh, "success_rate":count/500}
        return mu_va

    opt_var = parallel(lowhxf,mu_h,mu_x)
    csv(data=opt_var,file = path+"mu_x_vs_mu_h_vs_RCD.csv")
    graphicPlot(a= mu_h,b=mu_x ,array= opt_var,x_name=r'$\mu_x$',y_name=r"$\mu_h$",title="Number_of_options = 10",\
    save_name=path +"mu_x_vs_mu_h_vs_RCD.pdf",bar_label ='Rate of correct choice')


