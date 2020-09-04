# Nature of collective-decision making by simple 
# yes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

#%%

import numpy as np
import matplotlib.pyplot as plt
import classynu as yn
from multiprocessing import Pool
from operator import itemgetter 

#%%
number_of_options = 10                      #   Number of options to choose best one from
mu_x = 0.0                                  #   Mean of distribution from which quality stimulus are sampled randomly
sigma_x = 1.0                               #   Standard deviation of distribution from which quality stimulus are sampled randomly
mu_h = 0                                    #   Mean of distribution from which units threshold are sampled randomly
sigma_h = 1.0                               #   Standard deviation of distribution from which units threshold are sampled randomly
mu_m = 100                                  #   Mean of distribution from which number of units to be assigned to an option are sampled randomly
sigma_m = 0                                 #   Standard deviation of distribution from which number of units to be assigned to an option are sampled randomly
mu_assessment_err = 0.0                     #   Mean of distribution from which units quality assessment error are sampled randomly
sigma_assessment_err = 0.0                  #   Standard deviation of distribution from which units quality assessment error are sampled randomly
x_type = 0                                  #   Number of decimal places of quality stimulus
h_type = 3                                  #   Number of decimal places of units threshold
err_type = 0                                #   Number of decimal places of quality assessment error
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

def majority_decision(number_of_options,Dx,assigned_units,err_type,\
    mu_assessment_err,sigma_assessment_err,ref_highest_quality,\
        one_correct_opt = 1,quorum = None):
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
    one_correct_opt - (1,0) '0' means many correct options
    quorum - (int) quorum to be reached first in order to consider an option as best
    Returns:
    If quorum = None then success(1) or failure(0)
    else success(1) or failure(0) ,quorum_reached(success(1) or failure(0)),majority decision (one_correct(success(1) or failure(0)),multi_correct(success(1) or failure(0)))
    """
    DM = yn.Decision_making(number_of_options=number_of_options,err_type=err_type,\
    mu_assessment_err=mu_assessment_err,sigma_assessment_err=sigma_assessment_err)
    DM.quorum = quorum
    if quorum == None:
        DM.vote_counter(assigned_units,Dx)

        # plt.scatter(Dx,DM.votes)
        # plt.show()

        if one_correct_opt == 1:
            if DM.one_correct(ref_highest_quality) == 1:
                return 1
            else:
                return 0
            
        else:
            if DM.multi_correct(ref_highest_quality) == 1:
                return 1
            else:
                return 0
    else:
        correct,quorum_reached = DM.quorum_voting(assigned_units,Dx,ref_highest_quality)
        
        DM.vote_counter(assigned_units,Dx)
        one_correct = 0
        multi_correct = 0
        if one_correct_opt == 1:
            if DM.one_correct(ref_highest_quality) == 1:
                one_correct = 1
        else:
            if DM.multi_correct(ref_highest_quality) == 1:
                multi_correct = 1
        
        return correct,quorum_reached,one_correct,multi_correct

def one_run(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h,\
    x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err):

    pc = np.array(units(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m)).astype(int)

    units_distribution = []
    for i in pc:
        units_distribution.append(threshold(m_units = i ,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h))

    qc = quality(number_of_options=number_of_options,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x)

    dec = majority_decision(number_of_options=number_of_options,Dx = qc.Dx,assigned_units= units_distribution,\
        err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err,\
        ref_highest_quality=qc.ref_highest_quality)                                   

    if dec == 1:
        print("success")

    else:
        print("failed")

def multi_run(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h,\
    x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,err_type=err_type,mu_assessment_err= mu_assessment_err,\
    sigma_assessment_err=sigma_assessment_err,quorum= None):

    pc = np.array(units(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m)).astype(int)

    units_distribution = []
    for i in pc:
        units_distribution.append(threshold(m_units = i ,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h))

    qc = quality(number_of_options=number_of_options,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x)

    dec = majority_decision(number_of_options=number_of_options,Dx = qc.Dx,assigned_units= units_distribution,\
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
    plt.bar(quor,[1 for i in range(1,101,1)],width=1,color = "white",edgecolor='black')
    plt.bar(quor,list(map(itemgetter("q_not_reached"), opt_v)),width=1,color = "orange",edgecolor='black')
    plt.bar(quor,list(map(itemgetter("success_rate"), opt_v)),width=1,color = "blue",edgecolor='black')
    plt.plot(quorum,list(map(itemgetter(correct), opt_v)),color ="red")
    plt.xlabel('Quorum')
    plt.ylabel('Rate of choice')
    # plt.legend(title = leg)
    plt.savefig(save_name,format = "pdf")
    plt.show()

#%%
# Without assesment error Majority based decision
one_run()

#%%
# With assessment error Majority based decision
one_run(sigma_assessment_err=0.1)

#%%
# Random choice when more than one option's correct Majority based decision
one_run(x_type=0,err_type=0)

#%%
# Majority based Rate of correct choice as a function of sigma_h for varying number of options
sig_h = [0.0+i*0.01 for i in range(101)]
opts = [2,10]#2*i for i in range(2,6,3)]

def sighf(op,sigh):
    count = 0
    for k in range(2000):
        success = multi_run(sigma_h=sigh,number_of_options=op,err_type=0) 
        if success == 1:
            count += 1
    opt_va = {"opt":op,"sigma": sigh, "success_rate":count/2000}
    return opt_va

opt_var = parallel(sighf,opts,sig_h)

plt_show(data_len= opts,array= opt_var,var= "opt", plt_var="sigma",x_name='Sigma_h',\
    title="Number of options",save_name="Sigma_h_vs_Rate_of_correct_choice.pdf")

#%%
# Majority based Rate of correct choice as a function of mu_h for varying number of options 
m_h = [-4.0+i*0.08 for i in range(101)]
opts = [2,10]#2*i for i in range(2,6,3)]

def muhf(op,j):
    count = 0
    for k in range(2000):
        success = multi_run(mu_h=j,number_of_options=op,err_type=0) 
        if success == 1:
            count += 1
    opt_va = {"opt":op,"mu": j, "success_rate":count/2000}
    return opt_va

opt_var = parallel(muhf,opts,m_h)

plt_show(data_len= opts,array= opt_var,var= "opt", plt_var="mu",x_name='Mu_h',\
    title="Number of options",save_name="Mu_h_vs_Rate_of_correct_choice.pdf")

#%%
# Majority based Rate of correct choice as a function of number of options for varying mu_h 
number_of_options = [i for i in range(1,51,1)]
mu_h = [0,2]

def nf(muh,nop):
    count = 0
    for k in range(2000):
        success = multi_run(mu_h=muh,number_of_options=nop,err_type=0) 
        if success == 1:
            count += 1
    mu_h_va = {"nop":nop,"muh": muh, "success_rate":count/2000}
    return mu_h_va

opt_var = parallel(nf,mu_h,number_of_options)

plt_show(data_len= mu_h,array= opt_var,var= "muh", plt_var="nop",x_name='number_of_options',\
    title="mu_h",save_name="number_of_options_vs_Rate_of_correct_choice.pdf")

#%%
# Majority based Rate of correct choice as a function of mu_m for varying number of options 
mu_m = [i for i in range(1,101,1)]
number_of_options = [2,10]

def mumf(nop,mum):
    count = 0
    for k in range(2000):
        success = multi_run(mu_m=mum,number_of_options=nop,err_type=0) 
        if success == 1:
            count += 1
    nop_va = {"nop":nop,"mum": mum, "success_rate":count/2000}
    return nop_va

opt_var = parallel(mumf,number_of_options,mu_m)

plt_show(data_len= number_of_options,array= opt_var,var= "nop", plt_var="mum",x_name='number_of_units(variance = 0)',\
    title="Number_of_options",save_name="number_of_units_vs_Rate_of_correct_choice.pdf")

#%%
# Majority based Rate of correct choice as a function of sigma_m for varying number of options
sigma_m = [1+i*0.03 for i in range(0,1000,1)]
number_of_options = [2,10]

def sigmf(nop,sigm):
    count = 0
    for k in range(2000):
        success = multi_run(sigma_m=sigm,number_of_options=nop,err_type=0) 
        if success == 1:
            count += 1
    nop_va = {"nop":nop,"sigm": sigm, "success_rate":count/2000}
    return nop_vas

opt_var = parallel(sigmf,number_of_options,sigma_m)

plt_show(data_len= number_of_options,array= opt_var,var= "nop", plt_var="sigm",x_name='number_of_units',\
    title="Number_of_options",save_name="sigma_m_vs_rate_of_correct_choice.pdf")

#%%
# Majority based Rate of correct choice as a function of sigma_h for varying mu_h
sig_h = [np.round(0.0+i*0.01,decimals=2) for i in range(101)]
mu_h = [np.round(-4.0+i*0.08,decimals=2) for i in range(101)]

def sighf(mu,sig):
    count = 0
    for k in range(3000):
        success = multi_run(mu_h=mu,sigma_h=sig,err_type=0) 
        if success == 1:
            count += 1
    mu_va = {"mu":mu,"sigma": sig, "success_rate":count/3000}
    return mu_va

opt_var = parallel(sighf,mu_h,sig_h)

graphic_plt(a= mu_h,b=sig_h ,array= opt_var,x_name='Sigma_h',y_name="Mu_h",title="Number_of_options = 10",\
    save_name="mu_h_vs_sigma_h.pdf")

# %%
# Majority based Rate of correct choice as a function of mu_x for varying mu_h
mu_x = [np.round(-4.0+i*0.08,decimals=2) for i in range(101)]
mu_h = [np.round(-4.0+i*0.08,decimals=2) for i in range(101)]

def sighf(muh,mux):
    count = 0
    for k in range(3000):
        success = multi_run(mu_h=muh,mu_x=mux,err_type=0) 
        if success == 1:
            count += 1
    mu_va = {"mux":mux,"muh": muh, "success_rate":count/3000}
    return mu_va

opt_var = parallel(sighf,mu_h,mu_x)

graphic_plt(a= mu_h,b=mu_x ,array= opt_var,x_name='Mu_x',y_name="Mu_h",title="Number_of_options = 10",\
    save_name="mu_h_vs_mu_x.pdf")

# %%
# Majority based Rate of correct choice as a function of sigma_x for varying sigma_h
sig_x = [np.round(i*0.04,decimals=2) for i in range(101)]
sig_h = [np.round(i*0.04,decimals=2) for i in range(101)]

def sighf(sigh,sigx):
    count = 0
    for k in range(3000):
        success = multi_run(sigma_h=sigh,sigma_x=sigx,err_type=0) 
        if success == 1:
            count += 1
    sig_va = {"sigx":sigx,"sigh": sigh, "success_rate":count/3000}
    return sig_va

opt_var = parallel(sighf,sig_h,sig_x)

graphic_plt(a= sig_h,b=sig_x ,array= opt_var,x_name='Sigma_x',y_name="Sigma_h",title="Number_of_options = 10",\
    save_name="sig_h_vs_sig_x.pdf")

# %%
# Majority based Rate of choice as a function of quorum for varying sigma_m
quorum = [i for i in range(1,101,1)]
sig_m = [0,30]

def quof(sigm,quo):
    right_count = 0
    q_not_reached = 0
    one_correct_count = 0
    multi_correct_count = 0
    for k in range(2000):
        correct,quorum_reached,one_correct,multi_correct = multi_run(sigma_m=sigm,quorum=quo,err_type=0)
        right_count += correct
        if quorum_reached == 1:
            q_not_reached += 1
        multi_correct_count += multi_correct
        one_correct_count += one_correct
    sig_va = {"sigm":sigm,"quo": quo, "success_rate":right_count/2000,"q_not_reached":q_not_reached/2000,"one_correct":one_correct_count/2000,"multi_correct":multi_correct_count/2000}
    return sig_va

opt_var = parallel(quof,sig_m,quorum)

save_name = ["quorum_sigma_m"+str(i)+".pdf" for i in sig_m]
opt_v = {}
prev = None
for i in opt_var:
    if i["sigm"] == prev:
        opt_v[str(i["sigm"])].append(i)
        prev = i["sigm"]
    else:
        opt_v[str(i["sigm"])] = [i]
        prev = i["sigm"]

for i in range(len(save_name)):
    bar(quorum,opt_v[str(sig_m[i])],save_name[i],"one_correct")

# %%
# Decoy effect in individual decision and collective decision





