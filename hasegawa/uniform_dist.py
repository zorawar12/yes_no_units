# Nature of collective-decision making by simple yes/no decision units (Hasegawa paper).

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

#%%
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
if __name__ != "__main__":    
    import hasegawa.classynu as yn
else:
    import classynu as yn
from multiprocessing import Pool
from operator import itemgetter 


number_of_options = 10                      #   Number of options to choose best one from
low_x = -10.0                                  #   Mean of distribution from which quality stimulus are sampled randomly
high_x = 10.0                               #   Standard deviation of distribution from which quality stimulus are sampled randomly
low_h = -10.0                                    #   Mean of distribution from which units threshold are sampled randomly
high_h = 10.0                               #   Standard deviation of distribution from which units threshold are sampled randomly
low_m = 1                                 #   Mean of distribution from which number of units to be assigned to an option are sampled randomly
high_m = 1000                                 #   Standard deviation of distribution from which number of units to be assigned to an option are sampled randomly
low_assessment_err = -1.0                     #   Mean of distribution from which units quality assessment error are sampled randomly
high_assessment_err = 1.0                  #   Standard deviation of distribution from which units quality assessment error are sampled randomly
x_type = 3                                  #   Number of decimal places of quality stimulus
h_type = 3                                  #   Number of decimal places of units threshold
err_type = 0                                #   Number of decimal places of quality assessment error

Without_assesment_error_Majority_based_decision = 0
With_assesment_error_Majority_based_decision = 0
random_choice_many_best_option = 0
success_rate_low_h_number_options = 0
success_rate_high_h_number_options = 0
success_rate_number_options_low_h = 0
success_rate_low_m_number_options = 0
success_rate_high_m_number_options = 0
success_rate_high_h_low_h = 0
success_rate_mu_x_mu_h = 0
success_rate_sig_x_sig_h = 0
rate_of_choice_quorum_sig_m = 0


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
        # plt.scatter(Dx,DM.votes)
        # plt.show()

        return majority_dec
    
    else:
        result,quorum_reached = DM.quorum_voting(assigned_units,Dx,ref_highest_quality)
        return result,quorum_reached,majority_dec

def one_run(number_of_options=number_of_options,high_m=high_m,h_type=h_type,high_h=high_h,\
    x_type=x_type,high_x=sigma_x,err_type=err_type,high_assessment_err=high_assessment_err):

    pc = np.array(units(number_of_options=number_of_options,high_m=high_m)).astype(int)

    units_distribution = []
    for i in pc:
        units_distribution.append(threshold(m_units = i ,h_type=h_type,high_h=high_h))

    qc = quality(number_of_options=number_of_options,x_type=x_type,high_x=high_x)

    dec = majority_decision(number_of_options=number_of_options,Dx = qc.Dx,assigned_units= units_distribution,\
        err_type=err_type,high_assessment_err=high_assessment_err,\
        ref_highest_quality=qc.ref_highest_quality)

    if dec == 1:
        print("success")

    else:
        print("failed")

def multi_run(number_of_options=number_of_options,high_m=high_m,h_type=h_type,high_h=high_h,\
    x_type=x_type,high_x=high_x,err_type=err_type,high_assessment_err=high_assessment_err,quorum= None):

    pc = np.array(units(number_of_options=number_of_options,high_m=high_m)).astype(int)

    units_distribution = []
    for i in pc:
        units_distribution.append(threshold(m_units = i ,h_type=h_type,high_h=high_h))

    qc = quality(number_of_options=number_of_options,x_type=x_type,high_x=high_x)

    dec = majority_decision(number_of_options=number_of_options,Dx = qc.Dx,assigned_units= units_distribution,\
        err_type=err_type,high_assessment_err=high_assessment_err,\
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
    one_run(sigma_assessment_err=0.1)

# Random choice when more than one option's correct Majority based decision
if random_choice_many_best_option==1:
    one_run(x_type=0,err_type=0)

# Majority based Rate of correct choice as a function of sigma_h for varying number of options
if success_rate_sig_h_number_options==1:    
    sig_h = [0.0+i*0.01 for i in range(101)]
    opts = [2,10]#2*i for i in range(2,6,3)]

    def sighf(op,sigh):
        count = 0
        for k in range(1000):
            success = multi_run(sigma_h=sigh,number_of_options=op,err_type=0) 
            if success == 1:
                count += 1
        opt_va = {"opt":op,"sigma": sigh, "success_rate":count/1000}
        return opt_va

    opt_var = parallel(sighf,opts,sig_h)

    plt_show(data_len= opts,array= opt_var,var= "opt", plt_var="sigma",x_name='Sigma_h',\
        title="Number of options",save_name="na.pdf")

# Majority based Rate of correct choice as a function of mu_h for varying number of options 
if success_rate_mu_h_number_options==1:    
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

# Majority based Rate of correct choice as a function of number of options for varying mu_h 
if success_rate_number_options_mu_h==1:    
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

# Majority based Rate of correct choice as a function of mu_m for varying number of options 
if success_rate_mu_m_number_options==1:
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

# Majority based Rate of correct choice as a function of sigma_m for varying number of options
if success_rate_sig_m_number_options==1:
    sigma_m = [1+i*0.03 for i in range(0,1000,1)]
    number_of_options = [2,10]

    def sigmf(nop,sigm):
        count = 0
        for k in range(2000):
            success = multi_run(sigma_m=sigm,number_of_options=nop,err_type=0) 
            if success == 1:
                count += 1
        nop_va = {"nop":nop,"sigm": sigm, "success_rate":count/2000}
        return nop_va

    opt_var = parallel(sigmf,number_of_options,sigma_m)

    plt_show(data_len= number_of_options,array= opt_var,var= "nop", plt_var="sigm",x_name='number_of_units',\
        title="Number_of_options",save_name="sigma_m_vs_rate_of_correct_choice.pdf")

# Majority based Rate of correct choice as a function of sigma_h for varying mu_h
if success_rate_sig_h_mu_h==1:
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

# Majority based Rate of correct choice as a function of mu_x for varying mu_h
if success_rate_mu_x_mu_h==1:
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

    graphic_plt(a= mu_h,b=mu_x ,array= opt_var,x_name='Mu_x',y_name="Mu_h",title="Number_of_options = 10",\
        save_name="mu_h_vs_mu_x.pdf")

# Majority based Rate of correct choice as a function of sigma_x for varying sigma_h
if success_rate_sig_x_sig_h==1:
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

    graphic_plt(a= sig_h,b=sig_x ,array= opt_var,x_name='Sigma_x',y_name="Sigma_h",title="Number_of_options = 10",\
        save_name="sig_h_vs_sig_x.pdf")

# Majority based Rate of choice as a function of quorum for varying sigma_m
if rate_of_choice_quorum_sig_m==1:
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

    save_name = ["quorum_sigma_m"+str(i)+"unsorted.pdf" for i in sig_m]
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
        bar(quorum,opt_v[str(sig_m[i])],save_name[i],"maj")

# Decoy effect in individual decision and collective decision
