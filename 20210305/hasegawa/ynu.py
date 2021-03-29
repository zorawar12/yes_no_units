# Nature of collective-decision making by simple yes/no decision units (Hasegawa paper).

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import json
import classynu as yn
from multiprocessing import Pool
import time
# from ray.util.multiprocessing import Pool
import os
# import ray

# ray.init(address='auto', redis_password='5241590000000000')

path = os.getcwd() + "/results/"

Without_assesment_error_Majority_based_decision = 0
With_assesment_error_Majority_based_decision = 0
random_choice_many_best_option = 0
sig_h_vs_RCD_vs_nop = 0
mu_h_vs_RCD_vs_nop = 0
nop_vs_RCD_vs_mu_h = 0
mu_m_vs_RCD_nop = 0
sigma_m_vs_RCD_vs_nop = 0
mu_h_vs_sigma_h_vs_RCD = 0
mu_h_vs_mu_x_vs_RCD = 1
sig_h_vs_sig_x_vs_RCD = 0
quorum_vs_RC_vs_sig_m = 0

wf = yn.workFlow()
vis = yn.Visualization()

def parallel(func,a,b,batch_size,save_string,columns_name):
    f = open(path+save_string+'.csv','a')
    columns = pd.DataFrame(data=np.array([columns_name]))
    columns.to_csv(path+save_string+'.csv',mode='a',header=False,index=False)

    inp = []
    for i in a:
        for j in b:
            inp.append((i,j))

    opt_var = []
    progress = 0
    for i in range(0,len(inp),batch_size):
        t1 = time.time()
        with Pool(8) as p:#,ray_address="auto") as p:
            opt_var = p.starmap(func,inp[i:i+batch_size])
        t2 = time.time()
        print(t2-t1)
        
        # for i in inp:
        #     t1 = time.time()
        #     opt_var.append(func(i[0],i[1]))
        #     t2 = time.time()
        #     print(t2-t1)

        out = pd.DataFrame(data=opt_var)
        out.to_csv(path+save_string+'.csv',mode = 'a',header = False, index=False)
        progress +=1
        print("\r Percent of input processed : {}%".format(np.round(100*progress/(len(inp)/batch_size)),decimals=1), end="")


def save_data(save_string):
    check = np.sort(np.array([int(f) for f in os.listdir(path) if '.' not in f]))
    count = 0
    for i in check:
        if count==i:
            count+=1
    save_string = str(count)+save_string
    f1 = open(path+str(count),'w')
    return save_string

def pushbullet_message(title, body):
    msg = {"type": "note", "title": title, "body": body}
    TOKEN = 'o.YlTBKuQWnkOUsCP9ZxzWC9pvFNz1G0mi'
    resp = requests.post('https://api.pushbullet.com/v2/pushes', 
                         data=json.dumps(msg),
                         headers={'Authorization': 'Bearer ' + TOKEN,
                                  'Content-Type': 'application/json'})
    if resp.status_code != 200:
        raise Exception('Error',resp.status_code)
    else:
        print ('Message sent')

# Without assesment error Majority based decision
if Without_assesment_error_Majority_based_decision==1:
    number_of_options = 10          
    mu_x = 0.0          
    sigma_x = 1.0        
    mu_h = 0                           
    sigma_h = 1.0                   
    mu_m = 100  
    sigma_m = 0                         
    wf.one_run(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m,mu_h=mu_h,sigma_h=sigma_h,mu_x=mu_x,sigma_x=sigma_x)

# With assessment error Majority based decision
if With_assesment_error_Majority_based_decision==1:
    number_of_options = 10          
    mu_x = 0.0          
    sigma_x = 1.0        
    mu_h = 0                           
    sigma_h = 1.0                   
    mu_m = 100  
    sigma_m = 0                         
    wf.one_run(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m,mu_h=mu_h,sigma_h=sigma_h,mu_x=mu_x,sigma_x=sigma_x,sigma_assessment_err=0.1)

# Random choice when more than one option's correct Majority based decision
if random_choice_many_best_option==1:
    number_of_options = 10          
    mu_x = 0.0          
    sigma_x = 1.0        
    mu_h = 0                           
    sigma_h = 1.0                   
    mu_m = 100  
    sigma_m = 0                         
    wf.one_run(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m,mu_h=mu_h,sigma_h=sigma_h,mu_x=mu_x,x_type=0,sigma_x=sigma_x,sigma_assessment_err=0.1)

# Majority based Rate of correct choice as a function of sigma_h for varying number of options
if sig_h_vs_RCD_vs_nop==1:
    save_string='sig_h_vs_RCD_vs_nop'
    save_string = save_data(save_string)
    runs = 1000
    sig_h = [i*0.01 for i in range(501)]
    opts = [2,10,40]
    mu_m = 100
    sigma_m = 0
    mu_h = 0
    mu_x = 0.0
    sigma_x = 1

    def sighf(op,sigh):
        count = 0
        for k in range(runs):
            success = wf.multi_run(sigma_h_1=sigh,number_of_options=op,mu_m_1=mu_m,sigma_m_1=sigma_m,\
                mu_m_2=mu_m,sigma_m_2=sigma_m,mu_h_1=mu_h,mu_h_2=mu_h,sigma_h_2=sigh,mu_x_1=mu_x,sigma_x_1=sigma_x,mu_x_2=mu_x,sigma_x_2=sigma_x)
            if success == 1:
                count += 1
        opt_va = {'$opts$':op,'$\sigma_{h}$': sigh, 'success_rate':count/runs}
        return opt_va

    parallel(sighf,opts,sig_h,columns_name=['$\sigma_{h}$','$opts$','success_rate'],save_string=save_string,batch_size=5)
    
    vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,y_var_='$\sigma_{h}$',x_var_='$opts$',z_var_='success_rate',plot_type="line")


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
    save_string = "mu_h_vs_mu_x_vs_RCD"
    nop = [100]
    save_string = save_data(save_string)
    runs = 200
    mu_m = 100
    sigma_m = 0
    sigma_h = 1
    sigma_x = 1
    mu_x = [np.round(1+i*0.08,decimals=2) for i in range(101)]
    mu_h = [np.round(1+i*0.08,decimals=2) for i in range(101)]

    for n in range(len(nop)):
        save_string += str(nop[n])
        def sighf(muh,mux):
            count = 0
            for k in range(runs):
                success = wf.multi_run(sigma_h_1=sigma_h,number_of_options=nop[n],mu_m_1=mu_m,sigma_m_1=sigma_m,\
                    mu_m_2=mu_m,sigma_m_2=sigma_m,mu_h_1=muh,mu_h_2=muh,sigma_h_2=sigma_h,mu_x_1=mux,sigma_x_1=sigma_x,mu_x_2=mux,sigma_x_2=sigma_x)
                if success == 1:
                    count += 1
            mu_va = {"$\mu_{h}$":muh,"$\mu_{x}$": mux, "success_rate":count/runs}
            return mu_va

        parallel(sighf,mu_h,mu_x,columns_name=["$\mu_{h}$","$\mu_{x}$",'success_rate'],save_string=save_string,batch_size=len(mu_h))

        vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x}$',y_var_='$\mu_{h}$',cbar_orien="vertical",num_of_opts=nop[n],line_labels=[nop[n],nop[n]+1],z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x)
        save_string = save_string[:-1]
        message = str(nop[n])+'number of options simulation finished'
        pushbullet_message('Python Code','Results out!'+message)

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
