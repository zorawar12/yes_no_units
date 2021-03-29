# Nature of collective-decision making by simple yes/no decision units (Hasegawa paper).

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import classynu as yn
from multiprocessing import Pool
# from ray.util.multiprocessing import Pool
import os
import time
# import ray

# ray.init(address='auto', redis_password='5241590000000000')


mu_assessment_err = 0.0             #   Mean of distribution for units quality assessment error
sigma_assessment_err = 0.0          #   Standard deviation of distribution for units quality assessment error
x_type = 3                          #   Number of decimal places of quality stimulus
h_type = 3                          #   Number of decimal places of units threshold
err_type = 0                        #   Number of decimal places of quality assessment error
path = os.getcwd() + "/results/"


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

number_of_opts = [2]
mu_m_1=100
sigma_m_1=0
mu_m_2=100
sigma_m_2=0
sigma_h_1 = 1
sigma_h_2=1
sigma_x_1=1
sigma_x_2=1
runs = 500
batch_size = 50
delta_mu = 5

wf = yn.workFlow()
vis = yn.Visualization()

for nop in number_of_opts:
    number_of_options = nop
    save_string = "5delta_mu_5_mu_h_vs_mu_x1_mu_x2_vs_RCDnop2"#save_data('delta_mu_'+str(delta_mu)+'_mu_h_vs_mu_x1_mu_x2_vs_RCD'+'nop'+str(nop))

    # mu_x = [np.round(i*0.1,decimals=1) for i in range(151)]
    # mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]

    # def mux1muh1(muh,mux):
    #     mux1 = mux
    #     mux2 = delta_mu + mux
    #     muh1 = muh
    #     muh2 = muh
    #     count = 0
    #     for k in range(runs):
    #         success = wf.multi_run(mu_h_1=muh1,sigma_h_1=sigma_h_1,sigma_h_2=sigma_h_2,mu_h_2=muh2,\
    #         mu_x_1=mux1,mu_x_2=mux2,sigma_x_1=sigma_x_1,sigma_x_2=sigma_x_1,err_type=0,\
    #         number_of_options=number_of_options,mu_m_1=mu_m_1,sigma_m_1=sigma_m_1,mu_m_2=mu_m_2,\
    #         sigma_m_2=sigma_m_2)
    #         if success == 1:
    #             count += 1
    #     mu_va = {'$\mu_{h_1}$':muh1,'$\mu_{h_2}$':muh2,'$\mu_{x_1}$': mux1,'$\mu_{x_2}$': mux2,"success_rate":count/runs}
    #     return mu_va

    # parallel(mux1muh1,mu_h,mu_x,columns_name=['$\mu_{h_1}$','$\mu_{h_2}$','$\mu_{x_1}$','$\mu_{x_2}$',"success_rate"],save_string=save_string,batch_size=3*len(mu_h))

    vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=[number_of_options,number_of_options+1],z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2)

    message = str(nop)+' number of options simulation finished'
    pushbullet_message('Python Code','Results out! '+message)
