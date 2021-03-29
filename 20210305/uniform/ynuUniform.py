# Nature of collective-decision making by simple yes/no decision units (Hasegawa paper).

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

#%%
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
sigma_h_vs_RCD_vs_nop = 1
mu_h_vs_RCD_vs_nop = 0
nop_vs_RCD_vs_mu_h = 0
mu_m_vs_RCD_vs_nop = 0
sigma_m_vs_RCD_vs_nop = 0
sigma_h_vs_mu_h_vs_RCD = 0
mu_x_vs_mu_h_vs_RCD = 0
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

def linePlot(data_len,array,var,plt_var,x_name,y_name,title,save_name):
    c = ["blue","green","red","purple","brown","black"]
    count = 0
    fig = plt.figure()
    plt.style.use('ggplot')
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

# Without assesment error Majority based decision
if Without_assesment_error_Majority_based_decision==1:
    one_run()

# With assessment error Majority based decision
if With_assesment_error_Majority_based_decision==1:
    one_run(low_assessment_err=-0.1,high_assessment_err=0.1)

# Random choice when more than one option's correct Majority based decision
if random_choice_many_best_option==1:
    one_run(x_type=0,err_type=0)

# Majority based Rate of correct choice as a function of sigma_h for varying number of options
if sigma_h_vs_RCD_vs_nop==1:
    sig_h = [(0.0+i*0.01) for i in range(1001)]
    opts = [2,10,40,60]

    def sighf(op,sigh):
        count = 0
        for k in range(1000):
            success = multi_run(high_h=np.sqrt(3*sigh),low_h=-np.sqrt(3*sigh),number_of_options=op,err_type=0)
            if success == 1:
                count += 1
        opt_va = {"opt":op,"sigh": sigh, "success_rate":count/1000}
        return opt_va

    opt_var = parallel(sighf,opts,sig_h)
    csv(data=opt_var,file= path + "sigma_h_vs_RCD_nop_10.csv")
    linePlot(data_len= opts,array=opt_var,var= "opt", plt_var="sigh",x_name=r'$\sigma_h$',\
    y_name='Rate of correct choice',title="Number of options",save_name= path+"sigma_h_vs_RCD_vs_nop_10.pdf")

# Majority based Rate of correct choice as a function of mu_h for varying number of options
if mu_h_vs_RCD_vs_nop==1:
    mu_h = [-4.0+i*0.08 for i in range(101)]
    opts = [2,10]#2*i for i in range(2,6,3)]

    def lowhf(op,muh):
        count = 0
        for k in range(2000):
            success = multi_run(low_h=muh+low_h,high_h=muh+high_h,number_of_options=op,err_type=0) 
            if success == 1:
                count += 1
        opt_va = {"opt":op,"mu_h": muh, "success_rate":count/2000}
        return opt_va

    opt_var = parallel(lowhf,opts,mu_h)
    csv(data=opt_var,file= path+ "mu_h_vs_RCD_vs_nop.csv")
    linePlot(data_len= opts,array= opt_var,var= "opt", plt_var="mu_h",x_name=r'$\mu_h$',\
    y_name='Rate of correct choice',title="Number of options",save_name=path + "mu_h_vs_RCD_vs_nop.pdf")

# Majority based Rate of correct choice as a function of number of options for varying mu_h
if nop_vs_RCD_vs_mu_h==1:
    number_of_options = [i for i in range(1,51,1)]
    mu_h = [0,2,4]

    def nopf(muh,nop):
        count = 0
        for k in range(2000):
            success = multi_run(low_h=muh+low_h,high_h=muh+high_h,number_of_options=nop,err_type=0)
            if success == 1:
                count += 1
        mu_h_va = {"nop":nop,"muh": muh, "success_rate":count/2000}
        return mu_h_va

    opt_var = parallel(nopf,mu_h,number_of_options)
    csv(data=opt_var,file=path+ 'nop_vs_RCD_vs_mu_h.csv')
    linePlot(data_len= mu_h,array= opt_var,var= "muh", plt_var="nop",x_name='n',\
    y_name='Rate of correct choice',title=r"$\mu_h$",save_name=path+"nop_vs_RCD_vs_mu_h.pdf")

# Majority based Rate of correct choice as a function of mu_m for varying number of options
if mu_m_vs_RCD_vs_nop==1:
    mu_m = [i for i in range(1,101,1)]
    number_of_options = [2,10]

    def highmf(nop,mum):
        count = 0
        for k in range(2000):
            success = multi_run(high_m=mum+high_m,low_m=mum+low_m,number_of_options=nop,err_type=0)
            if success == 1:
                count += 1
        nop_va = {"nop":nop,"mum": mum, "success_rate":count/2000}
        return nop_va

    opt_var = parallel(highmf,number_of_options,mu_m)
    csv(data= opt_var,file = path+ "mu_m_vs_RCD_vs_nop.csv")
    linePlot(data_len= number_of_options,array= opt_var,var= "nop", plt_var="mum",x_name= r'$\mu_m$',\
    y_name='Rate of correct choice',title="Number_of_options",save_name=path+"mu_m_vs_RCD_vs_nop.pdf")

# Majority based Rate of correct choice as a function of sigma_m for varying number of options 
if sigma_m_vs_RCD_vs_nop==1:
    sigma_m = [i for i in range(1,101,1)]
    number_of_options = [2,10,40]

    def lowmf(nop,sigm):
        count = 0
        for k in range(2000):
            success = multi_run(low_m=-np.sqrt(3*sigm),high_m=np.sqrt(3*sigm),number_of_options=nop,err_type=0)
            if success == 1:
                count += 1
        nop_va = {"nop":nop,"sigm": sigm, "success_rate":count/2000}
        return nop_va

    opt_var = parallel(lowmf,number_of_options,sigma_m)
    csv(data=opt_var,file=path+"sigma_m_vs_RCD_vs_nop.csv")
    linePlot(data_len= number_of_options,array= opt_var,var= "nop", plt_var="sigm",x_name=r'$\sigma_m$',\
    y_name="Rate of correct choice",title="Number_of_options",save_name=path+"sigma_m_vs_RCD_vs_nop.pdf")

# Majority based Rate of correct choice as a function of sigma_h for varying mu_h
if sigma_h_vs_mu_h_vs_RCD==1:
    sigma_h = [np.round(0.0+i*0.01,decimals=2) for i in range(101)]
    mu_h = [np.round(-4.0+i*0.08,decimals=2) for i in range(101)]

    def highhf(muh,sigmah):
        count = 0
        for k in range(3000):
            success = multi_run(low_h=muh-np.sqrt(3*sigmah),high_h=muh+np.sqrt(3*sigmah),err_type=0)
            if success == 1:
                count += 1
        mu_va = {"muh":muh,"sigmah": sigmah, "success_rate":count/3000}
        return mu_va

    opt_var = parallel(highhf,mu_h,sigma_h)
    csv(data=opt_var,file=path+"sigma_h_vs_mu_h_vs_RCD.csv")
    graphicPlot(a= mu_h,b=sigma_h ,array= opt_var,x_name=r'$\sigma_h$',y_name=r"$\mu_h$",title="Number_of_options = 10",\
    save_name=path+"sigma_h_vs_mu_h_vs_RCD.pdf",bar_label="Rate of correct choice")

# Majority based Rate of correct choice as a function of mu_x for varying mu_h
if mu_x_vs_mu_h_vs_RCD==1:
    mu_x = [np.round(-4.0+i*0.08,decimals=2) for i in range(101)]
    mu_h = [np.round(-4.0+i*0.08,decimals=2) for i in range(101)]

    def lowhxf(muh,mux):
        count = 0
        for k in range(1000):
            success = multi_run(low_h=muh+low_h,low_x=mux+low_x,high_h=muh+high_h,high_x=mux+high_x,err_type=0)
            if success == 1:
                count += 1
        mu_va = {"mux":mux,"muh": muh, "success_rate":count/1000}
        return mu_va

    opt_var = parallel(lowhxf,mu_h,mu_x)
    csv(data=opt_var,file = path+"mu_x_vs_mu_h_vs_RCD.csv")
    graphicPlot(a= mu_h,b=mu_x ,array= opt_var,x_name=r'$\mu_x$',y_name=r"$\mu_h$",title="Number_of_options = 10",\
    save_name=path +"mu_x_vs_mu_h_vs_RCD.pdf",bar_label ='Rate of correct choice')

# Majority based Rate of correct choice as a function of sigma_x for varying sigma_h
if sigma_x_vs_sigma_h_vs_RCD==1:
    sigma_x = [np.round(i*0.04,decimals=2) for i in range(101)]
    sigma_h = [np.round(i*0.04,decimals=2) for i in range(101)]

    def highhf(sigh,sigx):
        count = 0
        for k in range(1000):
            success = multi_run(high_h=np.sqrt(3*sigh),high_x=np.sqrt(3*sigx),low_h=-np.sqrt(3*sigh),low_x=-np.sqrt(3*sigx),err_type=0)
            if success == 1:
                count += 1
        sig_va = {"sigx":sigx,"sigh": sigh, "success_rate":count/1000}
        return sig_va

    opt_var = parallel(highhf,sigma_h,sigma_x)
    csv(data=opt_var,file = path+ "sigma_x_vs_sigma_h_vs_RCD.csv")
    graphicPlot(a= sigma_h,b=sigma_x ,array= opt_var,x_name=r'$\sigma_x$',y_name=r"$\sigma_h$",title="Number_of_options = 10",\
    save_name=path+"sigma_x_vs_sigma_h_vs_RCD.pdf",bar_label="Rate of correct choice")

# Majority based Rate of choice as a function of quorum for varying sigma_m
if quorum_vs_RC_vs_sigma_m==1:
    quorum = [i for i in range(1,101,1)]
    sigma_m = [0,30]

    def quof(sigm,quo):
        right_count = 0
        q_not_reached = 0
        majority_dec_count = 0
        for k in range(1000):
            correct,quorum_reached,majority_dec = multi_run(high_m=np.sqrt(3*sigm),low_m=-np.sqrt(3*sigm),quorum=quo,err_type=0)
            right_count += correct
            if quorum_reached == 0:
                q_not_reached += 1
            majority_dec_count += majority_dec
        sig_va = {"sigm":sigm,"quo": quo, "success_rate":right_count/1000,"q_not_reached":q_not_reached/1000,"maj":majority_dec_count/1000}
        return sig_va

    opt_var = parallel(quof,sigma_m,quorum)
    csv(data=opt_var,file=path+"quorum_vs_RC_vs_sigma_m.csv")
    save_name = [path+"quorum_vs_RC_vs_sigma_m"+str(i)+".pdf" for i in sigma_m]
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
        barPlot(quorum,opt_v[str(sigma_m[i])],save_name[i],"maj")

# Decoy effect in individual decision and collective decision
