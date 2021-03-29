# Nature of collective-decision making by simple yes/no decision units (Hasegawa paper).

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

# import matplotlib
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import classynu as yn
from multiprocessing import Pool
# from ray.util.multiprocessing import Pool
# from operator import itemgetter
import os
import time
# import ray

# ray.init(address='auto', redis_password='5241590000000000')


mu_assessment_err = 0.0                     #   Mean of distribution from which units quality assessment error are sampled randomly
sigma_assessment_err = 0.0                  #   Standard deviation of distribution from which units quality assessment error are sampled randomly
x_type = 3                                  #   Number of decimal places of quality stimulus
h_type = 3                                  #   Number of decimal places of units threshold
err_type = 0                                #   Number of decimal places of quality assessment error
path = os.getcwd() + "/results/"

# def units(mu_m_1,sigma_m_1,mu_m_2,sigma_m_2,number_of_options):
#     a = np.array([])
#     peak_choice = np.random.randint(0,2,number_of_options)
#     for i in peak_choice:
#         if i==0:
#             k = np.round(np.random.normal(mu_m_1,sigma_m_1),decimals=0)
#             while k<=0:
#                 k = np.round(np.random.normal(mu_m_1,sigma_m_1),decimals=0)
#             a = np.append(a,k)
#         else:
#             k = np.round(np.random.normal(mu_m_2,sigma_m_2),decimals=0)
#             while k<=0:
#                 k = np.round(np.random.normal(mu_m_2,sigma_m_2),decimals=0)
#             a = np.append(a,k)
#     return a.astype(int)


# def threshold(m_units,h_type,mu_h_1,sigma_h_1,mu_h_2,sigma_h_2):
#     a = []
#     peak_choice = np.random.randint(0,2,m_units)
#     for i in peak_choice:
#         if i==0:
#             a.append(np.round(np.random.normal(mu_h_1,sigma_h_1),decimals=h_type))
#         else:
#             a.append(np.round(np.random.normal(mu_h_2,sigma_h_2),decimals=h_type))
#     return a

# def quality(number_of_options,x_type,mu_x_1,sigma_x_1,mu_x_2,sigma_x_2):
#     QC = yn.qualityControl(number_of_options=number_of_options,x_type=x_type)
#     QC.mu_x_1 = mu_x_1
#     QC.sigma_x_1 = sigma_x_1
#     QC.mu_x_2 = mu_x_2
#     QC.sigma_x_2 = sigma_x_2
#     QC.dx()
#     QC.ref_highest_qual()
#     return QC

# def majority_decision(number_of_options,Dx,assigned_units,err_type,\
#     mu_assessment_err,sigma_assessment_err,ref_highest_quality,\
#     quorum = None):

#     DM = yn.Decision_making(number_of_options=number_of_options,err_type=err_type,\
#     mu_assessment_err=mu_assessment_err,sigma_assessment_err=sigma_assessment_err)
#     DM.quorum = quorum
#     DM.vote_counter(assigned_units,Dx)
#     majority_dec = DM.best_among_bests(ref_highest_quality)

#     if quorum == None:
#         return majority_dec

#     else:
#         result,quorum_reached = DM.quorum_voting(assigned_units,Dx,ref_highest_quality)
#         return result,quorum_reached,majority_dec


# def multi_run(number_of_options=None,mu_m_1=None,sigma_m_1=None,\
#     mu_m_2=None,sigma_m_2=None,h_type=h_type,mu_h_1=None,sigma_h_1=None,mu_h_2=None,sigma_h_2=None,x_type=x_type,mu_x_1=None,sigma_x_1=None,mu_x_2=None,sigma_x_2=None,err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err,quorum= None):

#     pc = units(number_of_options=number_of_options,mu_m_1=mu_m_1,sigma_m_1=sigma_m_1,mu_m_2=mu_m_2,sigma_m_2=sigma_m_2)
#     # print(pc)
#     units_distribution = []
    
#     for i in pc:
#         units_distribution.append(threshold(m_units=i,h_type=h_type,mu_h_1=mu_h_1,sigma_h_1=sigma_h_1,mu_h_2=mu_h_2,sigma_h_2=sigma_h_2))
#     qc = quality(number_of_options=number_of_options,x_type=x_type,mu_x_1=mu_x_1,sigma_x_1=sigma_x_1,mu_x_2=mu_x_2,sigma_x_2=sigma_x_2)

#     dec = majority_decision(number_of_options=number_of_options,Dx = qc.Dx,assigned_units= units_distribution,\
#         err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err,\
#         ref_highest_quality=qc.ref_highest_quality,quorum=quorum)

#     return dec

# def parallel(func,a,b):
#     inp = []
#     for i in a:
#         for j in b:
#             inp.append((i,j))

#     # for inps in inp:
#     #     opt_var.append(func(inps[0],inps[1]))
#     opt_var = []
#     progress = 0
#     for i in range(0,len(inp),batch_size):
#         with Pool(12) as p:#,ray_address="auto") as p:
#             opt_var = p.starmap(func,inp[i:i+batch_size])
#         out = pd.DataFrame(data=opt_var)
#         out.to_csv(path+save_string+'.csv',mode = 'a',header = False, index=False)
#         progress +=1
#         print("\r Percent of input processed : {}%".format(np.round(100*progress/(len(inp)/batch_size)),decimals=1), end="")



# def linePlot(data_len,array,var,plt_var,x_name,y_name,title,save_name):
#     c = ["blue","green","red","purple","brown","black"]
#     count = 0
#     fig = plt.figure()
#     plt.style.use("ggplot")
#     data = [[] for i in range(len(data_len))]

#     for i in array:
#         data[data_len.index(i[var])].append(i)

#     for i in data:
#         plt.plot(list(map(itemgetter(plt_var), i)),list(map(itemgetter("success_rate"), i)),c = c[count],linewidth = 1)
#         count += 1

#     plt.xlabel(x_name)
#     plt.ylabel(y_name)
#     plt.legend(data_len,markerscale = 3, title = title)
#     plt.savefig(save_name,format = "pdf")
#     plt.show()

# def graphicPlot(a,b,array,x_name,y_name,z_name,title,save_name,cbar_loc,options_line,line_labels,z_var = None):
#     fig, ax = plt.subplots()
#     if z_var is not None:
#         z = np.array(z_var).reshape(len(a),len(b))
#     else:
#         z = np.array(list(map(itemgetter("success_rate"), array))).reshape(len(a),len(b))
#     cs = ax.pcolormesh(b,a,z)
#     colors = ["black","brown"]
    
#     for j in range(len(options_line)):
#         plt.plot(b,options_line[j],color = colors[j],linestyle='-.',label = str(line_labels[j]))
#     cbar = fig.colorbar(cs,orientation=cbar_loc)
#     cbar.set_label(z_name)
#     rec_low = max(a[0],b[0]) + 0.5
#     rec_high = min(a[-1],b[-1]) - 0.5
#     ax.plot([rec_low,rec_low],[rec_low,rec_high],color= 'red',linewidth = 0.5)
#     ax.plot([rec_low,rec_high],[rec_low,rec_low],color= 'red',linewidth = 0.5)
#     ax.plot([rec_high,rec_low],[rec_high,rec_high],color= 'red',linewidth = 0.5)
#     ax.plot([rec_high,rec_high],[rec_low,rec_high],color= 'red',linewidth = 0.5)
#     ax.set_aspect('equal', 'box')
#     plt.xlabel(x_name)
#     plt.ylabel(y_name + ' or options')
#     plt.legend()
#     plt.title(title)
#     plt.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
#     plt.minorticks_on()
#     plt.grid(b=True, which='minor', color='black', linestyle='-',linewidth = 0.2,alpha=0.1)
#     plt.savefig(save_name,format = "pdf")
#     plt.show()


# def csv(data,file):
#     f = pd.DataFrame(data=data)
#     f.to_csv(file)


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


# def data_visualize(file_name,save_plot,x_var_,y_var_,cbar_orien,line_labels,data =None,num_of_opts=None):
#     if data == None:
#         op = pd.read_csv(path+file_name)
#         opt_var = []

#         for j in range(len(op[x_var_])):
#             a = {}
#             for i in op:
#                 a[str(i)] = op[str(i)][j]
#             opt_var.append(a)
#     else:
#         opt_var = data
    
#     z_var_ = "success_rate"
#     x = []
#     y = []
#     z = []  # Make sure that it is in ordered form as y variable
#     for i in opt_var:
#         if i[x_var_] not in x:
#             x.append(i[x_var_])
#         if i[y_var_] not in y:
#             y.append(i[y_var_])
#         z.append(i[z_var_])

#         print(np.round(len(z)/len(opt_var),decimals=2),end="\r")
#     print(np.round(len(z)/len(opt_var),decimals=2))

#     x_1 = []
#     x_2 = []
#     for i in x:
#         [_1,intermediate_x,intermediate_fx] = finding_gaussian_base(1-(1/line_labels[0]),i,sigma_x_1)
#         [_2,intermediate_x,intermediate_fx] = finding_gaussian_base(1-(1/line_labels[1]),i,sigma_x_1)
#         x_1.append(_1)
#         x_2.append(_2)
    
#     graphicPlot(a= y,b=x,array= opt_var,x_name=r'%s'%x_var_,y_name=r'%s'%y_var_,z_name="Rate of correct choice",title="Number_of_options = "+str(num_of_opts),save_name=path+save_plot+x_var_[2:-1]+y_var_[2:-1]+'RCD.pdf',cbar_loc=cbar_orien,z_var=z,options_line=[x_1,x_2],line_labels=line_labels)

# def gaussian(x,mu,sigma):
#     k = 1/np.sqrt(2*np.pi*sigma)
#     return k*np.exp(-((x-mu)**2)/(2*sigma**2))

# def finding_gaussian_base(area,mu,sigma):
#     step = 0.00001
#     x = mu
#     x_ = []
#     fx_ = []
#     if area<0.5:
#         dummy_area = 0.5
#         while dummy_area-area >0.000:
#             x_.append(x)
#             fx = gaussian(x,mu,sigma)
#             fx_.append(fx)
#             dummy_area -= fx*step
#             x -= step
#         print(dummy_area)
#     elif area>0.5:
#         dummy_area = 0.5
#         while dummy_area-area <0.000:
#             x_.append(x)
#             fx = gaussian(x,mu,sigma)
#             fx_.append(fx)
#             dummy_area += fx*step
#             x += step
#         print(dummy_area)
#     else:
#         x_.append(x)
#         fx = gaussian(x,mu,sigma)
#         fx_.append(fx)
#         print(area)
#     return [x,x_,fx_]


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
    save_string = save_data('delta_mu_'+str(delta_mu)+'_mu_h_vs_mu_x1_mu_x2_vs_RCD'+'nop'+str(nop))

    mu_x = [np.round(i*0.1,decimals=1) for i in range(151)]
    mu_h = [np.round(i*0.1,decimals=1) for i in range(151)]

    def mux1muh1(muh,mux):
        mux1 = mux
        mux2 = delta_mu + mux
        muh1 = muh
        muh2 = muh
        count = 0
        for k in range(runs):
            success = wf.multi_run(mu_h_1=muh1,sigma_h_1=sigma_h_1,sigma_h_2=sigma_h_2,mu_h_2=muh2,mu_x_1=mux1,mu_x_2=mux2,sigma_x_1=sigma_x_1,sigma_x_2=sigma_x_1,err_type=0,number_of_options=number_of_options,mu_m_1=mu_m_1,sigma_m_1=sigma_m_1,mu_m_2=mu_m_2,sigma_m_2=sigma_m_2)
            if success == 1:
                count += 1
        mu_va = {'$\mu_{h_1}$':muh1,'$\mu_{h_2}$':muh2,'$\mu_{x_1}$': mux1,'$\mu_{x_2}$': mux2,"success_rate":count/runs}
        return mu_va

    parallel(mux1muh1,mu_h,mu_x,columns_name=['$\mu_{h_1}$','$\mu_{h_2}$','$\mu_{x_1}$','$\mu_{x_2}$',"success_rate"],save_string=save_string,batch_size=len(mu_h))


    vis.data_visualize(file_name=save_string+".csv",save_plot=save_string,x_var_='$\mu_{x_1}$',y_var_='$\mu_{h_1}$',cbar_orien="vertical",num_of_opts=nop,line_labels=[number_of_options,number_of_options+1],z_var_='success_rate',plot_type='graphics',sigma_x_1=sigma_x_1,delta_mu=delta_mu,sigma_x_2=sigma_x_2)

    message = str(nop)+' number of options simulation finished'
    pushbullet_message('Python Code','Results out! '+message)


check_gaussian = 1
if check_gaussian == 0:
    # [_,p_x,p_fx] = finding_gaussian_base(1.000000000,5,1)
    # [_,n_x,n_fx] = finding_gaussian_base(0.0000000,5,1)
    # [_,intermediate_x,intermediate_fx] = finding_gaussian_base(0.1,5,1)
    # x = [n_x[i] for i in range(len(n_x)-1,-1,-1)]+p_x
    # fx = [n_fx[i] for i in range(len(n_fx)-1,-1,-1)]+p_fx
    # f,ax = plt.subplots()
    # plt.plot(x,fx)
    # x = np.array(x)[:x.index(intermediate_x[-1])]
    # fx = np.array(fx)[:fx.index(intermediate_fx[-1])]
    # ax.fill_between(x,0,fx,facecolor='orange')
    # plt.show()
    number_of_options = [2,5,10,20,100]
    f,ax = plt.subplots(5,5)
    for nop in number_of_options:
        for l in range(5):
            options_quality = quality(number_of_options=nop,x_type=x_type,mu_x_1=5,sigma_x_1=1,mu_x_2=8,sigma_x_2=1)
            options_quality = options_quality.Dx
            y = [1 for i in range(len(options_quality))]
            [_,p1_x,p1_fx] = finding_gaussian_base(1.000000000,5,1)
            [_,n1_x,n1_fx] = finding_gaussian_base(0.0000000,5,1)
            [_,p2_x,p2_fx] = finding_gaussian_base(1.000000000,8,1)
            [_,n2_x,n2_fx] = finding_gaussian_base(0.0000000,8,1)
            x = [n1_x[i] for i in range(len(n1_x)-1,-1,-1)]+p1_x
            fx = [n1_fx[i] for i in range(len(n1_fx)-1,-1,-1)]+p1_fx 
            ax[number_of_options.index(nop),l].plot(x,fx)
            x = [n2_x[i] for i in range(len(n2_x)-1,-1,-1)]+p2_x
            fx = [n2_fx[i] for i in range(len(n2_fx)-1,-1,-1)]+p2_fx
            ax[number_of_options.index(nop),l].plot(x,fx)
            for i in range(len(y)):
                ax[number_of_options.index(nop),l].plot([options_quality[i],options_quality[i]],[0,1])
    plt.show()

