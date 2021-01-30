# Nature of collective-decision making by simple yes/no decision units (Hasegawa paper).

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

#%%
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
if __name__ != "__main__":
    import hasegawa.classynu as yn
else:
    import classynu as yn
# from multiprocessing import Pool
from ray.util.multiprocessing import Pool
from operator import itemgetter
import os
import csv as c
import ray

ray.init(address='auto', redis_password='5241590000000000')

number_of_options = 10                      #   Number of options to choose best one from
mu_x = 0.0                                  #   Mean of distribution from which quality stimulus are sampled randomly
sigma_x = 1.0                               #   Standard deviation of distribution from which quality stimulus are sampled randomly
mu_h = 0                                    #   Mean of distribution from which units threshold are sampled randomly
sigma_h = 1.0                               #   Standard deviation of distribution from which units threshold are sampled randomly
mu_m = 100                                  #   Mean of distribution from which number of units to be assigned to an option are sampled randomly
sigma_m = 0                                 #   Standard deviation of distribution from which number of units to be assigned to an option are sampled randomly
mu_assessment_err = 0.0                     #   Mean of distribution from which units quality assessment error are sampled randomly
sigma_assessment_err = 0.0                  #   Standard deviation of distribution from which units quality assessment error are sampled randomly
x_type = 3                                  #   Number of decimal places of quality stimulus
h_type = 3                                  #   Number of decimal places of units threshold
err_type = 0                                #   Number of decimal places of quality assessment error
path = os.getcwd() + "/results/"

Without_assesment_error_Majority_based_decision = 0
With_assesment_error_Majority_based_decision = 0
random_choice_many_best_option = 0
sig_h_vs_RCD_vs_nop = 1
mu_h_vs_RCD_vs_nop = 0
nop_vs_RCD_vs_mu_h = 0
mu_m_vs_RCD_nop = 0
sigma_m_vs_RCD_vs_nop = 0
mu_h_vs_sigma_h_vs_RCD = 0
mu_h_vs_mu_x_vs_RCD = 0
sig_h_vs_sig_x_vs_RCD = 0
quorum_vs_RC_vs_sig_m = 0

#@ray.remote
def units(mu_m,sigma_m,number_of_options):
    """
    Arguments:
    mu_m -(int) mean of distribution from where to choose number of units to be assigned
    sigma_m - (int) standard deviation of distribution from where to choose number of units to be assigned
    Returns:
    Number of units to be assigned to an option choosen from N(mu_m,sigma_m) (array[1xn])
    """
    a = np.array(np.round(np.random.normal(mu_m,sigma_m,number_of_options),decimals=0))
    for i in range(len(a)):
        while a[i] <=0:
            a[i] = np.array(np.round(np.random.normal(mu_m,sigma_m,number_of_options),decimals=0))
    return a

#@ray.remote
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

#@ray.remote
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

#@ray.remote
def majority_decision(number_of_options,Dx,assigned_units,err_type,\
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
    majority_dec = DM.best_among_bests(ref_highest_quality)
    if quorum == None:
        # plt.scatter(Dx,DM.votes)
        # plt.show()

        return majority_dec

    else:
        result,quorum_reached = DM.quorum_voting(assigned_units,Dx,ref_highest_quality)
        return result,quorum_reached,majority_dec

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

#@ray.remote
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

def parallel(func,a,b):
    inp = []
    for i in a:
        for j in b:
            inp.append((i,j))

    opt_var = []

    with Pool(24,ray_address="auto") as p:
        opt_var = p.starmap(func,inp)

    return opt_var



def linePlot(data_len,array,var,plt_var,x_name,y_name,title,save_name):
    c = ["blue","green","red","purple","brown","black"]
    count = 0
    fig = plt.figure()
    plt.style.use("ggplot")
    data = [[] for i in range(len(data_len))]

    for i in array:
        data[data_len.index(i[var])].append(i)

    for i in data:
        plt.plot(list(map(itemgetter(plt_var), i)),list(map(itemgetter("success_rate"), i)),c = c[count],linewidth = 1)
        count += 1

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend(data_len,markerscale = 3, title = title)
    plt.savefig(save_name,format = "pdf")
    plt.show()

def graphicPlot(a,b,array,x_name,y_name,z_name,title,save_name):
    fig, ax = plt.subplots()
    z = np.array(list(map(itemgetter("success_rate"), array))).reshape(len(a),len(b))
    cs = ax.contourf(b,a,z)
    cbar = fig.colorbar(cs)
    cbar.set_label(z_name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend(title = title)
    plt.savefig(save_name,format = "pdf")
    plt.show()

def barPlot(quor,opt_v,save_name,correct):
    fig, ax = plt.subplots()
    ax.bar(quor,[1 for i in range(1,101,1)],width=1,color = "white",edgecolor='black')
    ax.bar(quor,list(map(itemgetter("success_rate"), opt_v)),width=1,color = "blue",edgecolor='black')
    ax.bar(quor,list(map(itemgetter("q_not_reached"), opt_v)),bottom=list(map(itemgetter("success_rate"), opt_v)),width=1,color = "orange",edgecolor='black')
    plt.plot(quorum,list(map(itemgetter(correct), opt_v)),color ="red")
    plt.xlabel('Quorum size')
    plt.ylabel('Rate of choice')
    plt.savefig(save_name,format = "pdf")
    plt.show()

def csv(data,file):
    f = pd.DataFrame(data=data)
    f.to_csv(file)

def save_data(save_string):
    check = sorted(os.listdir(path))
    count = 0
    for i in check:
        if str(count)+'.txt'==i:
            count+=1
    save_string = str(count)+save_string
    open(path+str(count)+'.txt','+w')
    return save_string
# Without assesment error Majority based decision
if Without_assesment_error_Majority_based_decision==1:
    one_run()

# With assessment error Majority based decision
if With_assesment_error_Majority_based_decision==1:
    one_run(sigma_assessment_err=0.1)

# Random choice when more than one option's correct Majority based decision
if random_choice_many_best_option==1:
    one_run(x_type=0,err_type=0)

# Majority based Rate of correct choice as a function of sigma_h for varying number of options
if sig_h_vs_RCD_vs_nop==1:
    save_string = save_data('9999mu_h_vs_RCD_vs_nop')
    f = open(path+save_string+'.csv','a')
    columns = pd.DataFrame(data=np.array([["opt","sigma","success_rate"]]))
    columns.to_csv(path+save_string+'.csv',mode='a',header=False,index=False)
    sig_h = [0.0+i*0.01 for i in range(501)]
    opts = [2,10,40]

    # @ray.remote
    def sighf(op,sigh):
        global f
        count = 0
        for k in range(1000):
            success = multi_run(sigma_h=sigh,number_of_options=op,err_type=0)
            if success == 1:
                count += 1
        opt_va = {"opt":op,"sigma": sigh, "success_rate":count/1000}
        out = np.array([[op,sigh,count/1000]])
        out = pd.DataFrame(data=out)
        out.to_csv(path+save_string+'.csv',mode='a',header=False,index=False)

        return opt_va

    opt_var = parallel(sighf,opts,sig_h)
    csv(data=opt_var,file =path +save_string +'last.csv')
    linePlot(data_len= opts,array= opt_var,var= "opt", plt_var="sigma",\
    x_name=r'$\sigma_h$',y_name="Rate of correct choice", title="Number of options",\
    save_name=path + save_string+".pdf")
    # opt_var = parallel(sighf,opts,sig_h)
    # csv(data=opt_var,file =path +save_string +'last.csv')
    # linePlot(data_len= opts,array= opt_var,var= "opt", plt_var="sigma",\
    # x_name=r'$\sigma_h$',y_name="Rate of correct choice", title="Number of options",\
    # save_name=path + save_string+".pdf")

# Majority based Rate of correct choice as a function of mu_h for varying number of options
if mu_h_vs_RCD_vs_nop==1:
    m_h = [-4.0+i*0.08 for i in range(101)]
    opts = [2,10]#2*i for i in range(2,6,3)]

    def muhf(op,j):
        count = 0
        for k in range(000):
            success = multi_run(mu_h=j,number_of_options=op,err_type=0)
            if success == 1:
                count += 1
        opt_va = {"opt":op,"mu": j, "success_rate":count/2000}
        return opt_va

    opt_var = parallel(muhf,opts,m_h)
    csv(data=opt_var,file=path+"mu_h_vs_RCD_vs_nop.csv")
    linePlot(data_len= opts,array= opt_var,var= "opt", plt_var="mu",x_name=r'$\mu_h$',\
    y_name="Rate of correct choice" ,title="Number of options",save_name=path +"mu_h_vs_RCD_vs_nop.pdf")

# Majority based Rate of correct choice as a function of number of options for varying mu_h
if nop_vs_RCD_vs_mu_h==1:
    number_of_options = [i for i in range(1,51,1)]
    mu_h = [0,2,4,6]

    def nf(muh,nop):
        count = 0
        for k in range(2000):
            success = multi_run(mu_h=muh,number_of_options=nop,err_type=0)
            if success == 1:
                count += 1
        mu_h_va = {"nop":nop,"muh": muh, "success_rate":count/2000}
        return mu_h_va

    opt_var = parallel(nf,mu_h,number_of_options)
    csv(data = opt_var,file=path+"nop_vs_RCD_vs_mu_h.csv")
    linePlot(data_len= mu_h,array= opt_var,var= "muh", plt_var="nop",x_name='n',\
    y_name = "Rate of correct choice",title=r"$\mu_h$",save_name=path+"nop_vs_RCD_vs_mu_h.pdf")

# Majority based Rate of correct choice as a function of mu_m for varying number of options 
if mu_m_vs_RCD_nop==1:
    mu_m = [i for i in range(1,101,1)]
    number_of_options = [2,10,40,60]

    def mumf(nop,mum):
        count = 0
        for k in range(2000):
            success = multi_run(mu_m=mum,number_of_options=nop,err_type=0)
            if success == 1:
                count += 1
        nop_va = {"nop":nop,"mum": mum, "success_rate":count/2000}
        return nop_va

    opt_var = parallel(mumf,number_of_options,mu_m)
    csv(data=opt_var,file=path+"mu_m_vs_RCD_nop.csv")
    linePlot(data_len= number_of_options,array= opt_var,var= "nop", plt_var="mum",x_name=r'$\mu_m$',\
    y_name = "Rate of correct choice",title="Number_of_options",save_name=path+"mu_m_vs_RCD_nop.pdf")

# Majority based Rate of correct choice as a function of sigma_m for varying number of options
if sigma_m_vs_RCD_vs_nop==1:
    sigma_m = [1+i*0.03 for i in range(0,1000,1)]
    number_of_options = [2,10,40,60]

    def sigmf(nop,sigm):
        count = 0
        for k in range(2000):
            success = multi_run(sigma_m=sigm,number_of_options=nop,err_type=0)
            if success == 1:
                count += 1
        nop_va = {"nop":nop,"sigm": sigm, "success_rate":count/2000}
        return nop_va

    opt_var = parallel(sigmf,number_of_options,sigma_m)
    csv(data = opt_var,file=path+"sigma_m_vs_RCD_vs_nop.csv")
    linePlot(data_len= number_of_options,array= opt_var,var= "nop", plt_var="sigm",x_name=r'$\sigma_m$',\
    y_name = "Rate of correct choice",title="Number_of_options",save_name=path + "sigma_m_vs_RCD_vs_nop.pdf")

# Majority based Rate of correct choice as a function of sigma_h for varying mu_h
if mu_h_vs_sigma_h_vs_RCD==1:
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
    csv(data=opt_var,file=path+"mu_h_vs_sigma_h_vs_RCD.csv")
    graphicPlot(a= mu_h,b=sig_h ,array= opt_var,x_name='$\sigma_h$',y_name=r"$\mu_h$",\
    z_name="Rate of correct choice",title="Number_of_options = 10",\
    save_name=path+"mu_h_vs_sigma_h_vs_RCD.pdf")

# Majority based Rate of correct choice as a function of mu_x for varying mu_h
if mu_h_vs_mu_x_vs_RCD==1:
    mu_x = [np.round(-4.0+i*0.08,decimals=2) for i in range(101)]
    mu_h = [np.round(-4.0+i*0.08,decimals=2) for i in range(101)]

    def sighf(muh,mux):
        count = 0
        for k in range(1000):
            success = multi_run(mu_h=muh,mu_x=mux,err_type=0)
            if success == 1:
                count += 1
        mu_va = {"mux":mux,"muh": muh, "success_rate":count/1000}
        return mu_va

    opt_var = parallel(sighf,mu_h,mu_x)
    csv(data=opt_var,file=path+"mu_h_vs_mu_x_vs_RCD.csv")
    graphicPlot(a= mu_h,b=mu_x ,array= opt_var,x_name=r'$\mu_x$',y_name=r"$\mu_h$",z_name="Rate of correct choice",\
    title="Number_of_options = 10",save_name=path+"mu_h_vs_mu_x_vs_RCD.pdf")

# Majority based Rate of correct choice as a function of sigma_x for varying sigma_h
if sig_h_vs_sig_x_vs_RCD==1:
    sig_x = [np.round(i*0.04,decimals=2) for i in range(101)]
    sig_h = [np.round(i*0.04,decimals=2) for i in range(101)]

    def sighf(sigh,sigx):
        count = 0
        for k in range(1000):
            success = multi_run(sigma_h=sigh,sigma_x=sigx,err_type=0) 
            if success == 1:
                count += 1
        sig_va = {"sigx":sigx,"sigh": sigh, "success_rate":count/1000}
        return sig_va

    opt_var = parallel(sighf,sig_h,sig_x)
    csv(data=opt_var,file=path+"sig_h_vs_sig_x_vs_RCD.csv")
    graphicPlot(a= sig_h,b=sig_x ,array= opt_var,x_name=r'$\sigma_x$',y_name=r"$\sigma_h$",z_name="Rate of correct choice",\
    title="Number_of_options = 10",save_name=path+"sig_h_vs_sig_x_vs_RCD.pdf")

# Majority based Rate of choice as a function of quorum for varying sigma_m
if quorum_vs_RC_vs_sig_m==1:
    quorum = [i for i in range(1,101,1)]
    sig_m = [0,30]

    def quof(sigm,quo):
        right_count = 0
        q_not_reached = 0
        majority_dec_count = 0
        for k in range(1000):
            correct,quorum_reached,majority_dec = multi_run(sigma_m=sigm,quorum=quo,err_type=0)
            right_count += correct
            if quorum_reached == 0:
                q_not_reached += 1
            majority_dec_count += majority_dec
        sig_va = {"sigm":sigm,"quo": quo, "success_rate":right_count/1000,"q_not_reached":q_not_reached/1000,"maj":majority_dec_count/1000}
        return sig_va

    opt_var = parallel(quof,sig_m,quorum)
    csv(data=opt_var,file=path+"quorum_vs_RC_vs_sig_m.csv")
    save_name = [path+"quorum_sigma_m"+str(i)+"unsorted.pdf" for i in sig_m]
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
        barPlot(quorum,opt_v[str(sig_m[i])],save_name[i],"maj")

# Decoy effect in individual decision and collective decision
