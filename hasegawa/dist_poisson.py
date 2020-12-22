# Nature of collective-decision making by simple yes/no decision units (Hasegawa paper).

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

#%%
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import poisson_class as yn
from multiprocessing import Pool
from operator import itemgetter


number_of_options = 10
lambda_x = 5.0
lambda_h = 5.0
lambda_m = 5.0
lambda_assessment_err = 1.0
x_type = 3                 
h_type = 3
err_type = 0

Without_assesment_error_Majority_based_decision = 0
With_assesment_error_Majority_based_decision = 0
random_choice_many_best_option = 0
success_rate_low_h_number_options = 0
success_rate_high_h_number_options = 0
success_rate_number_options_low_h = 0
success_rate_low_m_number_options = 0
success_rate_high_m_number_options = 0
success_rate_high_h_low_h = 0
success_rate_low_x_low_h = 0
success_rate_high_x_high_h = 0
rate_of_choice_quorum_high_m = 0


def units(lambda_m,number_of_options):
    return np.round(np.random.poisson(lam=lambda_m,size=number_of_options),decimals=0)

def threshold(m_units,h_type,lambda_h):
    return np.round(np.random.poisson(lambda_h,size=abs(m_units)),decimals=h_type)

def quality(number_of_options,x_type,lambda_x):
    QC = yn.qualityControl(number_of_options=number_of_options,x_type=x_type)
    QC.lambda_x = lambda_x
    QC.dx()
    QC.ref_highest_qual()
    return QC

def majority_decision(number_of_options,Dx,assigned_units,err_type,\
    lambda_assessment_err,ref_highest_quality,quorum = None):

    DM = yn.Decision_making(number_of_options=number_of_options,err_type=err_type,\
    lambda_assessment_err=lambda_assessment_err)
    DM.quorum = quorum
    DM.vote_counter(assigned_units,Dx)
    majority_dec = DM.best_among_bests(ref_highest_quality)
    if quorum == None:
        plt.scatter(Dx,DM.votes)
        plt.show()

        return majority_dec

    else:
        result,quorum_reached = DM.quorum_voting(assigned_units,Dx,ref_highest_quality)
        return result,quorum_reached,majority_dec

def one_run(number_of_options=number_of_options,lambda_m=lambda_m,h_type=h_type,lambda_h=lambda_h,x_type=x_type,lambda_x=lambda_x,err_type=err_type,lambda_assessment_err=lambda_assessment_err):

    pc = np.array(units(number_of_options=number_of_options,lambda_m=lambda_m)).astype(int)

    units_distribution = []
    for i in pc:
        units_distribution.append(threshold(m_units = i ,h_type=h_type,lambda_h=lambda_h))

    qc = quality(number_of_options=number_of_options,x_type=x_type,lambda_x=lambda_x)

    dec = majority_decision(number_of_options=number_of_options,Dx = qc.Dx,assigned_units= units_distribution,\
        err_type=err_type,lambda_assessment_err=lambda_assessment_err,\
        ref_highest_quality=qc.ref_highest_quality)

    if dec == 1:
        print("success")

    else:
        print("failed")

def multi_run(number_of_options=number_of_options,lambda_m=lambda_m,h_type=h_type,lambda_h=lambda_h,\
    x_type=x_type,lambda_x=lambda_x,err_type=err_type,lambda_assessment_err=lambda_assessment_err,quorum= None):

    pc = np.array(units(number_of_options=number_of_options,lambda_m=lambda_m)).astype(int)

    units_distribution = []
    for i in pc:
        units_distribution.append(threshold(m_units = i ,h_type=h_type,low_h=low_h,high_h=high_h))

    qc = quality(number_of_options=number_of_options,x_type=x_type,lambda_x=lambda_x)

    dec = majority_decision(number_of_options=number_of_options,Dx = qc.Dx,assigned_units= units_distribution,\
        err_type=err_type,lambda_assessment_err=lambda_assessment_err,\
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
        plt.scatter(list(map(itemgetter(plt_var), i)),list(map(itemgetter("success_rate"), i)),c = c[count],s=3)
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

    with Pool(20) as p:
        opt_var = p.starmap(func,inp)

    return opt_var

def graphic_plt(a,b,array,x_name,y_name,title,save_name,bar_label):
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

def bar(quor,opt_v,save_name,correct):
    fig, ax = plt.subplots()
    ax.bar(quor,[1 for i in range(1,101,1)],width=1,color = "white",edgecolor='black')
    ax.bar(quor,list(map(itemgetter("success_rate"), opt_v)),width=1,color = "blue",edgecolor='black')
    ax.bar(quor,list(map(itemgetter("q_not_reached"), opt_v)),bottom=list(map(itemgetter("success_rate"), opt_v)),width=1,color = "orange",edgecolor='black')
    plt.plot(quorum,list(map(itemgetter(correct), opt_v)),color ="red")
    plt.xlabel('Quorum')
    plt.ylabel('Rate of choice')
    # plt.legend(title = leg)
    plt.savefig(save_name,format = "pdf")
    plt.show()

# Without assesment error Majority based decision
if Without_assesment_error_Majority_based_decision==1:
    one_run()

# With assessment error Majority based decision
if With_assesment_error_Majority_based_decision==1:
    one_run(lambda_assessment_err=1)

# Random choice when more than one option's correct Majority based decision
if random_choice_many_best_option==1:
    one_run(x_type=0,err_type=0)

# Majority based Rate of correct choice as a function of sigma_h for varying number of options
if success_rate_high_h_number_options==1:
    high_h = [(0.0+i*0.01) for i in range(101)]
    opts = [2*i for i in range(2,6,3)]

    def highf(op,highh):
        count = 0
        for k in range(1000):
            success = multi_run(high_h=highh,number_of_options=op,err_type=0)
            if success == 1:
                count += 1
        opt_va = {"opt":op,"highh": highh, "success_rate":count/1000}
        return opt_va

    opt_var = parallel(highf,opts,high_h)

    plt_show(data_len= opts,array= opt_var,var= "opt", plt_var="highh",x_name=r'$high_h$',\
        title="Number of options",save_name="high_h_vs_RCD_vs_nop.pdf")

# Majority based Rate of correct choice as a function of mu_h for varying number of options 
if success_rate_low_h_number_options==1:
    low_h = [-4.0+i*0.08 for i in range(101)]
    opts = [2,10]#2*i for i in range(2,6,3)]

    def lowhf(op,lowh):
        count = 0
        for k in range(2000):
            success = multi_run(low_h=lowh,number_of_options=op,err_type=0) 
            if success == 1:
                count += 1
        opt_va = {"opt":op,"low_h": lowh, "success_rate":count/2000}
        return opt_va

    opt_var = parallel(lowhf,opts,low_h)

    plt_show(data_len= opts,array= opt_var,var= "opt", plt_var="low_h",x_name=r'$low_h$',\
        title="Number of options",save_name="Low_h_vs_RCD_vs_nop.pdf")

# Majority based Rate of correct choice as a function of number of options for varying mu_h 
if success_rate_number_options_low_h==1:
    number_of_options = [i for i in range(1,51,1)]
    low_h = [0,2]

    def nopf(lowh,nop):
        count = 0
        for k in range(2000):
            success = multi_run(low_h=lowh,number_of_options=nop,err_type=0)
            if success == 1:
                count += 1
        mu_h_va = {"nop":nop,"lowh": lowh, "success_rate":count/2000}
        return mu_h_va

    opt_var = parallel(nopf,low_h,number_of_options)

    plt_show(data_len= low_h,array= opt_var,var= "lowh", plt_var="nop",x_name='Number_of_options',\
        title=r"$low_h$",save_name="nop_vs_RCD_vs_low_h.pdf")

# Majority based Rate of correct choice as a function of mu_m for varying number of options 
if success_rate_low_m_number_options==1:
    low_m = [i for i in range(1,101,1)]
    number_of_options = [2,10]

    def lowmf(nop,lowm):
        count = 0
        for k in range(2000):
            success = multi_run(low_m=lowm,number_of_options=nop,err_type=0)
            if success == 1:
                count += 1
        nop_va = {"nop":nop,"lowm": lowm, "success_rate":count/2000}
        return nop_va

    opt_var = parallel(lowmf,number_of_options,low_m)

    plt_show(data_len= number_of_options,array= opt_var,var= "nop", plt_var="lowm",x_name=r'$low_m$',\
        title="Number_of_options",save_name="low_m_vs_RCD_vs_nop.pdf")

# Majority based Rate of correct choice as a function of sigma_m for varying number of options
if success_rate_high_m_number_options==1:
    high_m = [1+i*0.03 for i in range(0,1000,1)]
    number_of_options = [2,10]

    def highmf(nop,highm):
        count = 0
        for k in range(2000):
            success = multi_run(high_m=highm,number_of_options=nop,err_type=0) 
            if success == 1:
                count += 1
        nop_va = {"nop":nop,"highm": highm, "success_rate":count/2000}
        return nop_va

    opt_var = parallel(highmf,number_of_options,high_m)

    plt_show(data_len= number_of_options,array= opt_var,var= "nop", plt_var="highm",x_name=r'$high_m$',\
        title="Number_of_options",save_name="high_m_vs_RCD_vs_nop.pdf")

# Majority based Rate of correct choice as a function of sigma_h for varying mu_h
if success_rate_high_h_low_h==1:
    high_h = [np.round(0.0+i*0.01,decimals=2) for i in range(101)]
    low_h = [np.round(-4.0+i*0.08,decimals=2) for i in range(101)]

    def highhf(lowh,highh):
        count = 0
        for k in range(3000):
            success = multi_run(low_h=lowh,high_h=highh,err_type=0)
            if success == 1:
                count += 1
        mu_va = {"lowh":lowh,"highh": highh, "success_rate":count/3000}
        return mu_va

    opt_var = parallel(highhf,low_h,high_h)

    graphic_plt(a= low_h,b=high_h ,array= opt_var,x_name=r'$high_h$',y_name=r"$low_h$",title="Number_of_options = 10",\
        save_name="low_h_vs_high_h_vs_RCD.pdf",bar_label="Rate of correct choice")

# Majority based Rate of correct choice as a function of mu_x for varying mu_h
if success_rate_low_x_low_h==1:
    low_x = [np.round(-4.0+i*0.08,decimals=2) for i in range(101)]
    low_h = [np.round(-4.0+i*0.08,decimals=2) for i in range(101)]

    def lowhxf(lowh,lowx):
        count = 0
        for k in range(1000):
            success = multi_run(low_h=lowh,low_x=lowx,err_type=0)
            if success == 1:
                count += 1
        mu_va = {"lowx":lowx,"lowh": lowh, "success_rate":count/1000}
        return mu_va

    opt_var = parallel(lowhxf,low_h,low_x)

    graphic_plt(a= low_h,b=low_x ,array= opt_var,x_name=r'$low_x$',y_name=r"$low_h$",title="Number_of_options = 10",\
        save_name="low_h_vs_low_x_vs_RCD.pdf",bar_label ='Rate of correct choice')

# Majority based Rate of correct choice as a function of sigma_x for varying sigma_h
if success_rate_high_x_high_h==1:
    high_x = [np.round(i*0.04,decimals=2) for i in range(101)]
    high_h = [np.round(i*0.04,decimals=2) for i in range(101)]

    def highhf(highh,highx):
        count = 0
        for k in range(1000):
            success = multi_run(high_h=highh,high_x=highx,err_type=0)
            if success == 1:
                count += 1
        sig_va = {"highx":highx,"highh": highh, "success_rate":count/1000}
        return sig_va

    opt_var = parallel(highhf,high_h,high_x)

    graphic_plt(a= high_h,b=high_x ,array= opt_var,x_name=r'$high_x$',y_name=r"$high_h$",title="Number_of_options = 10",\
        save_name="high_h_vs_high_x.pdf",bar_label="Rate of correct choice")

# Majority based Rate of choice as a function of quorum for varying sigma_m
if rate_of_choice_quorum_high_m==1:
    quorum = [i for i in range(1,101,1)]
    high_m = [0,30]

    def quof(highm,quo):
        right_count = 0
        q_not_reached = 0
        majority_dec_count = 0
        for k in range(1000):
            correct,quorum_reached,majority_dec = multi_run(high_m=highm,quorum=quo,err_type=0)
            right_count += correct
            if quorum_reached == 0:
                q_not_reached += 1
            majority_dec_count += majority_dec
        sig_va = {"highm":highm,"quo": quo, "success_rate":right_count/1000,"q_not_reached":q_not_reached/1000,"maj":majority_dec_count/1000}
        return sig_va

    opt_var = parallel(quof,high_m,quorum)

    save_name = ["quorum_high_m"+str(i)+".pdf" for i in high_m]
    opt_v = {}
    prev = None
    for i in opt_var:
        if i["highm"] == prev:
            opt_v[str(i["highm"])].append(i)
            prev = i["highm"]
        else:
            opt_v[str(i["highm"])] = [i]
            prev = i["highm"]

    for i in range(len(save_name)):
        bar(quorum,opt_v[str(high_m[i])],save_name[i],"maj")

# Decoy effect in individual decision and collective decision
