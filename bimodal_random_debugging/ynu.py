# Nature of collective-decision making by simple yes/no decision units (Hasegawa paper).

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

#%%
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import classynu as yn
from multiprocessing import Pool
# from ray.util.multiprocessing import Pool
from operator import itemgetter
import os
# import ray
from numba import jit,njit,vectorize,guvectorize
# ray.init(address='auto', redis_password='5241590000000000')
import time

# number_of_options = None                      #   Number of options to choose best one from
# mu_x_1 = 0.0                                  #   Mean of distribution from which quality stimulus are sampled randomly
# sigma_x_1 = 1.0                               #   Standard deviation of distribution from which quality stimulus are sampled randomly
# mu_x_2 = 1.0                                  #   Mean of distribution from which quality stimulus are sampled randomly
# sigma_x_2 = 1.0                               #   Standard deviation of distribution from which quality stimulus are sampled randomly
# mu_h_1 = 0                                    #   Mean of distribution from which units threshold are sampled randomly
# sigma_h_1 = 1.0                               #   Standard deviation of distribution from which units threshold are sampled randomly
# mu_h_2 = 1                                    #   Mean of distribution from which units threshold are sampled randomly
# sigma_h_2 = 1.0                               #   Standard deviation of distribution from which units threshold are sampled randomly
# mu_m_1 = 100                                  #   Mean of distribution from which number of units to be assigned to an option are sampled randomly
# sigma_m_1 = 0                                 #   Standard deviation of distribution from which number of units to be assigned to an option are sampled randomly
# mu_m_2 = 100                                  #   Mean of distribution from which number of units to be assigned to an option are sampled randomly
# sigma_m_2 = 0                                 #   Standard deviation of distribution from which number of units to be assigned to an option are sampled randomly
mu_assessment_err = 0.0                     #   Mean of distribution from which units quality assessment error are sampled randomly
sigma_assessment_err = 0.0                  #   Standard deviation of distribution from which units quality assessment error are sampled randomly
x_type = 3                                  #   Number of decimal places of quality stimulus
h_type = 3                                  #   Number of decimal places of units threshold
err_type = 0                                #   Number of decimal places of quality assessment error
path = os.getcwd() + "/results/"


mu_h_vs_mu_x_vs_RCD = 1


def units(mu_m_1,sigma_m_1,mu_m_2,sigma_m_2,number_of_options):
    a = np.array([])
    peak_choice = np.random.randint(0,2,number_of_options)
    for i in peak_choice:
        if i==0:
            k = np.round(np.random.normal(mu_m_1,sigma_m_1),decimals=0)
            while k<=0:
                k = np.round(np.random.normal(mu_m_1,sigma_m_1),decimals=0)
            a = np.append(a,k)
        else:
            k = np.round(np.random.normal(mu_m_2,sigma_m_2),decimals=0)
            while k<=0:
                k = np.round(np.random.normal(mu_m_2,sigma_m_2),decimals=0)
            a = np.append(a,k)
    return a.astype(int)


#%%
def threshold(m_units,h_type,mu_h_1,sigma_h_1,mu_h_2,sigma_h_2):
    a = []
    peak_choice = np.random.randint(0,2,m_units)
    for i in peak_choice:
        if i==0:
            a.append(np.round(np.random.normal(mu_h_1,sigma_h_1),decimals=h_type))
        else:
            a.append(np.round(np.random.normal(mu_h_2,sigma_h_2),decimals=h_type))
    return a

def quality(number_of_options,x_type,mu_x_1,sigma_x_1,mu_x_2,sigma_x_2):
    QC = yn.qualityControl(number_of_options=number_of_options,x_type=x_type)
    QC.mu_x_1 = mu_x_1
    QC.sigma_x_1 = sigma_x_1
    QC.mu_x_2 = mu_x_2
    QC.sigma_x_2 = sigma_x_2
    QC.dx()
    QC.ref_highest_qual()
    return QC

def majority_decision(number_of_options,Dx,assigned_units,err_type,\
    mu_assessment_err,sigma_assessment_err,ref_highest_quality,\
    quorum = None):

    DM = yn.Decision_making(number_of_options=number_of_options,err_type=err_type,\
    mu_assessment_err=mu_assessment_err,sigma_assessment_err=sigma_assessment_err)
    DM.quorum = quorum
    DM.vote_counter(assigned_units,Dx)
    majority_dec = DM.best_among_bests(ref_highest_quality)

    if quorum == None:
        return majority_dec

    else:
        result,quorum_reached = DM.quorum_voting(assigned_units,Dx,ref_highest_quality)
        return result,quorum_reached,majority_dec


def multi_run(number_of_options=None,mu_m_1=None,sigma_m_1=None,\
    mu_m_2=None,sigma_m_2=None,h_type=h_type,mu_h_1=None,sigma_h_1=None,mu_h_2=None,sigma_h_2=None,x_type=x_type,mu_x_1=None,sigma_x_1=None,mu_x_2=None,sigma_x_2=None,err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err,quorum= None):
    
    pc = units(number_of_options=number_of_options,mu_m_1=mu_m_1,sigma_m_1=sigma_m_1,mu_m_2=mu_m_2,sigma_m_2=sigma_m_2)

    units_distribution = []
    for i in pc:
        units_distribution.append(threshold(m_units=i,h_type=h_type,mu_h_1=mu_h_1,sigma_h_1=sigma_h_1,mu_h_2=mu_h_2,sigma_h_2=sigma_h_2))

    qc = quality(number_of_options=number_of_options,x_type=x_type,mu_x_1=mu_x_1,sigma_x_1=sigma_x_1,mu_x_2=mu_x_2,sigma_x_2=sigma_x_2)

    dec = majority_decision(number_of_options=number_of_options,Dx = qc.Dx,assigned_units= units_distribution,\
        err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err,\
        ref_highest_quality=qc.ref_highest_quality,quorum=quorum)

    return dec

def parallel(func,a,b):
    inp = []
    for i in a:
        for j in b:
            inp.append((i,j))

    # for inps in inp:
    #     opt_var.append(func(inps[0],inps[1]))
    opt_var = []
    main_output = []
    progress = 0
    for i in range(0,len(inp),20):
        with Pool(20) as p:#,ray_address="auto") as p:
            opt_var = p.starmap(func,inp[i:i+20])
        main_output += opt_var
        out = pd.DataFrame(data=opt_var)
        out.to_csv(path+save_string+'.csv',mode = 'a',header = False, index=False)
        progress +=1
        print(100*progress/(len(inp)/20))
    return main_output

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

def graphicPlot(a,b,array,x_name,y_name,z_name,title,save_name,cbar_loc,z_var = None):
    fig, ax = plt.subplots()
    if z_var is not None:
        z = np.array(z_var).reshape(len(a),len(b))
    else:
        z = np.array(list(map(itemgetter("success_rate"), array))).reshape(len(a),len(b))
    cs = ax.contourf(b,a,z)
    cbar = fig.colorbar(cs,orientation=cbar_loc)
    cbar.set_label(z_name)
    rec_low = max(a[0],b[0]) + 0.5
    rec_high = min(a[-1],b[-1]) - 0.5
    ax.plot([rec_low,rec_low],[rec_low,rec_high],color= 'red',linewidth = 0.5)
    ax.plot([rec_low,rec_high],[rec_low,rec_low],color= 'red',linewidth = 0.5)
    ax.plot([rec_high,rec_low],[rec_high,rec_high],color= 'red',linewidth = 0.5)
    ax.plot([rec_high,rec_high],[rec_low,rec_high],color= 'red',linewidth = 0.5)
    ax.set_aspect('equal', 'box')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='black', linestyle='-',linewidth = 0.2,alpha=0.1)
    plt.savefig(save_name,format = "pdf")
    plt.show()


def csv(data,file):
    f = pd.DataFrame(data=data)
    f.to_csv(file)

def save_data(save_string):
    check = np.sort(os.listdir(path))
    count = 0
    for i in check:
        if str(count)==i:
            count+=1
    save_string = str(count)+save_string
    f1 = open(path+str(count),'w')
    return save_string


def data_visualize(file_name,save_plot,x_var_,y_var_,cbar_orien,data =[],num_of_opts=number_of_options):
    if data == []:
        op = pd.read_csv(path+file_name)
        opt_var = []

        for j in range(len(op[x_var_])):
            a = {}
            for i in op:
                a[str(i)] = op[str(i)][j]
            opt_var.append(a)
    else:
        opt_var = data
    
    z_var_ = "success_rate"
    x = []
    y = []
    z = []  # Make sure that it is in ordered form as y variable
    for i in opt_var:
        if i[x_var_] not in x:
            x.append(i[x_var_])
        if i[y_var_] not in y:
            y.append(i[y_var_])
        z.append(i[z_var_])

        print(np.round(len(z)/len(opt_var),decimals=2),end="\r")
    print(np.round(len(z)/len(opt_var),decimals=2))
    
    graphicPlot(a= y,b=x ,array= opt_var,x_name=r'%s'%x_var_,y_name=r'%s'%y_var_,z_name="Rate of correct choice",title="Number_of_options = "+str(num_of_opts),save_name=path+save_plot+x_var_[2:-1]+y_var_[2:-1]+'RCD.pdf',cbar_loc=cbar_orien,z_var=z)


# Majority based Rate of correct choice as a function of mu_x for varying mu_h
if mu_h_vs_mu_x_vs_RCD==1:
    number_of_opts = [2]
    mu_m_1=100
    sigma_m_1=0
    mu_m_2=100
    sigma_m_2=0
    for nop in number_of_opts:
        number_of_options = nop
        save_string = save_data('delta_mu_5_mu_h_1_mu_h_2_vs_mu_x1_mu_x2_vs_RCD'+'nop'+str(nop))
        f = open(path+save_string+'.csv','a')
        column = pd.DataFrame(data = np.array([['$\mu_{h_1}$','$\mu_{h_2}$','$\mu_{x_1}$','$\mu_{x_2}$',"success_rate"]]))
        column.to_csv(path+save_string+'.csv',mode='a',header= False,index=False)
        mu_x = [np.round(1+i*0.1,decimals=1) for i in range(5)]
        mu_h = [np.round(1+i*0.1,decimals=1) for i in range(5)]

        def mux1muh1(muh,mux):
            mux1 = mux
            mux2 = 2+mux
            muh1 = muh
            muh2 = 2+muh
            count = 0
            for k in range(1000):
                success = multi_run(mu_h_1=muh1,mu_h_2=muh2,mu_x_1=mux1,mu_x_2=mux2,err_type=0,number_of_options=number_of_options,mu_m_1=mu_m_1,sigma_m_1=sigma_m_1,mu_m_2=mu_m_2,sigma_m_2=sigma_m_2)
                if success == 1:
                    count += 1
            mu_va = {'$\mu_{h_1}$':muh1,'$\mu_{h_2}$':muh2,'$\mu_{x_1}$': mux1,'$\mu_{x_2}$': mux2,"success_rate":count/1000}
            return mu_va

        opt_var1 = parallel(mux1muh1,mu_h,mu_x)

        data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=nop)

