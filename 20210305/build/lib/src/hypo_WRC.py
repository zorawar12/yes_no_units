# Nature of collective-decision making by simple yes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in
import csv as c
import numpy as np
import matplotlib.pyplot as plt
if __name__ != "__main__":
    import src.classexploration as yn
else:
    import classexploration as yn
from multiprocessing import Pool
from operator import itemgetter
import pandas as pd
import os

# Accounting for no units along with yes units

number_of_options = 10                      #   Number of options to choose best one from
mu_x = 0.0                                  #   Mean of distribution from which quality stimulus are sampled randomly
sigma_x = 1.0                               #   Standard deviation of distribution from which quality stimulus are sampled randomly
mu_h = 0                                    #   Mean of distribution from which units threshold are sampled randomly
sigma_h = 1.0                               #   Standard deviation of distribution from which units threshold are sampled randomly
mu_m = 100                                  #   Mean of distribution from which number of units to be assigned to an option are sampled randomly
sigma_m = 100                                #   Standard deviation of distribution from which number of units to be assigned to an option are sampled randomly
mu_assessment_err = 0.0                     #   Mean of distribution from which units quality assessment error are sampled randomly
sigma_assessment_err = 0.0                  #   Standard deviation of distribution from which units quality assessment error are sampled randomly
x_type = 3                                  #   Number of decimal places of quality stimulus
h_type = 3                                  #   Number of decimal places of units threshold
err_type = 0                                #   Number of decimal places of quality assessment error
confidence = 0.02                           #   Confidence for distinguishing qualities
path = os.getcwd() + "/results/"

wrong_ranking_cost_contour = 0
pval_WRC_2D = 1

def units(mu_m,sigma_m,number_of_options):
    a = np.array(np.round(np.random.normal(mu_m,sigma_m,number_of_options),decimals=0))
    for i in range(len(a)):
        while a[i]<=0:
            a[i] = np.round(np.random.normal(mu_m,sigma_m),decimals=0)
    return a

def threshold(m_units,h_type,mu_h,sigma_h):
    return np.round(np.random.normal(mu_h,sigma_h,abs(m_units)),decimals=h_type)

def quality(number_of_options,x_type,mu_x,sigma_x):
    QC = yn.qualityControl(number_of_options=number_of_options,x_type=x_type)
    QC.mu_x = mu_x
    QC.sigma_x = sigma_x
    QC.dx()
    QC.ref_highest_qual()
    return QC

def decision_make_check(number_of_options,Dx,assigned_units,err_type,mu_assessment_err,sigma_assessment_err,\
    ref_highest_quality,pc = None,quorum = None):

    DM = yn.Decision_making(number_of_options=number_of_options,err_type=err_type,\
    mu_assessment_err=mu_assessment_err,sigma_assessment_err=sigma_assessment_err)
    DM.quorum = quorum
    DM.vote_counter(assigned_units,Dx)
    DM.for_against_vote_counter(assigned_units,Dx,pc)
    majority_dec = DM.best_among_bests_no(ref_highest_quality)
    qrincorrectness = yn.Qranking(number_of_options)
    qrincorrectness.ref_ranking(Dx,DM.y_ratios,DM.no_votes)
    incorrectness = qrincorrectness.incorrectness_cost(qrincorrectness.exp_rank)
    qrincorrectness.ref_ranking_w_n(Dx,DM.y_ratios,DM.no_votes)
    incorrectness_w_n = qrincorrectness.incorrectness_cost(qrincorrectness.exp_rank_w_n)
    if quorum == None:
        # plt.scatter(Dx,DM.votes)
        # plt.show()
        return majority_dec,incorrectness,incorrectness_w_n,DM.yes_stats,DM.max_ratio_pvalue
    else:
        result,quorum_reached = DM.quorum_voting(assigned_units,Dx,ref_highest_quality)
        return result,quorum_reached,majority_dec

def main_process_flow(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h,\
    x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,err_type=err_type,mu_assessment_err= mu_assessment_err,\
    sigma_assessment_err=sigma_assessment_err,quorum= None):

    pc = np.array(units(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m)).astype(int)

    units_distribution = []
    for i in pc:
        units_distribution.append(threshold(m_units = i ,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h))

    qc = quality(number_of_options=number_of_options,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x)

    dec = decision_make_check(number_of_options=number_of_options,Dx = qc.Dx,assigned_units= units_distribution,\
        err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err,\
        ref_highest_quality=qc.ref_highest_quality,quorum=quorum,pc = pc)

    return dec

def parallel(func,a,b):
    inp = []
    for i in a:
        for j in b:
            inp.append((i,j))

    opt_var = []

    with Pool(16) as p:
        opt_var = p.starmap(func,inp)

    return opt_var

def parallel3(func,a,b,c):
    inp = []
    for i in a:
        for j in b:
            for k in c:
                inp.append((i,j,k))

    opt_var = []

    with Pool(16) as p:
        opt_var = p.starmap(func,inp)

    return opt_var

def graphicPlot(a,b,zvar,array,x_name,y_name,title,save_name,bar_label):
    fig, ax = plt.subplots()
    z = np.array(list(map(itemgetter(zvar), array))).reshape(len(a),len(b))
    cs = ax.contourf(b,a,z)
    cbar = fig.colorbar(cs)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    cbar.set_label(bar_label)
    ax.legend(title = title)
    plt.savefig(save_name,format = "pdf")
    plt.show()

def linePlot(data_len,array,var,plt_var,x_name,y_name,title,save_name,y_var,data_legend):
    c = ["blue","green","red","purple","brown","yellow","black","orange","pink"]
    line_style = ["-","--",":","-."]
    count = 0
    fig = plt.figure(figsize=(15, 8), dpi= 90, facecolor='w', edgecolor='k')
    plt.style.use('ggplot')
    data = [[] for i in range(len(data_len))]

    for i in array:
        data[data_len.index(i[var])].append(i)

    for j in y_var:
        for i in data:
            plt.plot(list(map(itemgetter(plt_var), i)),list(map(itemgetter(j), i))\
            ,c = c[count%len(c)],ls = line_style[count%len(line_style)])
            count += 1
    plt.legend(data_legend,markerscale = 1, title = title)
    plt.ylim(top = 0.3,bottom = -0.1)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.savefig(save_name,format = "pdf")
    plt.show()

def csv(data,file):
    f = pd.DataFrame(data=data)
    f.to_csv(file)

def save_data(save_string):
    check = os.listdir(path)
    count = 0
    for i in check:
        if str(count)+'.txt'==i:
            count+=1
    save_string = str(count) + save_string
    f1 = open(path+str(count)+'.txt','w')
    return save_string

if __name__ != "__main__":
    # This is cor checking confidence on 'yes' counts among each pair of options
    sig_h = [0.0+i*0.01 for i in range(101)]
    opts = [5,10]

    def sighf(op,sigh):
        count = 0
        for k in range(1000):
            success,yes_test,_,_,_ = main_process_flow(sigma_h=sigh,number_of_options=op,err_type=0)
            print(yes_test)
            if success == 1:
                count += 1
        return {"opt":op,"sigma": sigh, "success_rate":count/1000}

    output = parallel(sighf,opts,sig_h)
    csv(data= output,file = path+"unknown.csv")
    linePlot(data_len= opts,array= output,var= "opt", plt_var="sigma",x_name=r'$\sigma_h$',\
    y_name = "Rate of correct choice",y_var = ["success_rate"],data_legend = opts,title="Number of options",save_name=path+ "Sigma_h_vs_Rate_of_correct_choice_sorted_no.pdf")

if wrong_ranking_cost_contour==1:
    save_string = '1wrong_ranking_cost_contour' #save_data("wrong_ranking_cost_contour")
#    f = open(path+save_string+'.csv','a')
#    columns = pd.DataFrame(data = np.array([["sigm","mum", "success_rate","avg_pval","avg_incrt", "avg_incrt_w_n","avg_correct_ranking"]]))
#    columns.to_csv(path+save_string+'.csv',mode='a',header= False,index=False)

    mu_m = [i for i in range(500,1000)]
    sigma_m = [i for i in range(0,200)]
    number_of_options = 10

#    def mumf(mum,sigm):
#        count = 0
#        sum_pval = 0
#        avg_incrtness = 0
#        avg_incrtness_w_n = 0
#        avg_correct_ranking = 0
#        loop = 0
#        while loop <= 1000:
#            success,incrt,incrt_w_n,yes_test,max_rat_pval = main_process_flow(mu_m=mum,sigma_m=sigm,err_type=0)
#            flag = 0
#            for i in yes_test:
#                for j in i:
#                    if j[0][0]== np.nan or j[1]<0:
#                        flag = 1
#                        break
#            if max_rat_pval[0][0]!= np.nan and max_rat_pval[1]>0 and flag!=1:
#                sum_pval += max_rat_pval[0][1]
#            else:
#                sum_pval += 1
#
#            avg_incrtness += incrt
#            avg_incrtness_w_n += incrt_w_n
#
#            if incrt_w_n == 0:
#                avg_correct_ranking +=1
#            if success == 1:
#                count += 1
#            loop+=1
#
#        avg_pval = sum_pval/1000
#        avg_incrtness = avg_incrtness/1000
#        avg_incrtness_w_n = avg_incrtness_w_n/1000
#        avg_correct_ranking = avg_correct_ranking/1000
#        v = {"sigm":sigm,"mum": mum, "success_rate":count/1000,'avg_pval':avg_pval,'avg_incrt':avg_incrtness, 'avg_incrt_w_n':avg_incrtness_w_n,'avg_correct_ranking':avg_correct_ranking}
#        out = np.array([[sigm, mum, count/1000,avg_pval,avg_incrtness, avg_incrtness_w_n,avg_correct_ranking]])
#        out = pd.DataFrame(data=out)
#        out.to_csv(path+save_string+'.csv',mode='a',header= False,index = False)
#        return v
#
#    opt_var = parallel(mumf,mu_m,sigma_m)
#    csv(data = opt_var,file = path+save_string+'last.csv')
    op = pd.read_csv(path+'1wrong_ranking_cost_contourlast.csv')
    opt_var = []

    for j in range(len(op['mum'])):
        a = {}
        for i in op:
            a[str(i)] = op[str(i)][j]
        opt_var.append(a)

    graphicPlot(a=mu_m,b=sigma_m,zvar="avg_incrt",array=opt_var,x_name=r"$\sigma_m$",y_name=r"$\mu_m$",title='number of options = '+str(number_of_options),save_name=path+save_string+"with_no.pdf",bar_label='Wrong ranking cost')
    graphicPlot(a=mu_m,b=sigma_m,zvar="avg_correct_ranking",array=opt_var,x_name = r"$\sigma_m$",y_name=r"$\mu_m$",title="number of options = "+str(number_of_options),save_name=path+save_string+"RCR.pdf",bar_label='Rate of correct ranking')
    graphicPlot(a=mu_m,b=sigma_m,zvar="avg_incrt_w_n",array=opt_var,x_name = r"$\sigma_m$",y_name=r"$\mu_m$",title="number of options = "+str(number_of_options),save_name=path+save_string+"without_no.pdf",bar_label='Wrong ranking cost')


if pval_WRC_2D ==1:
    save_string = save_data('pval_WRC_2D')
    f = open(path + save_string+'.csv','a')
    column = pd.DataFrame(data = np.array([["nop","mum","success_rate",'avg_pval','avg_incrt', 'avg_incrt_w_n']]))
    column.to_csv(path+save_string+'.csv',mode='a',header=False,index=False)

    mu_m = [i for i in range(500,2000,20)]
    number_of_options = [2,5,10,20]

    def mumf(nop,mum):
        count = 0
        sum_pval = 0
        avg_incrtness = 0
        avg_incrtness_w_n = 0
        loop = 0
        while loop<=2000:
            success,incrt,incrt_w_n,yes_test,max_rat_pval = main_process_flow(mu_m=mum,number_of_options=nop,err_type=0)
            flag = 0
            for i in yes_test:
                for j in i:
                    if j[0][0]== np.nan or j[1]<0:
                        flag = 1
                        break
            if max_rat_pval[0][0]!= np.nan and max_rat_pval[1]>0 and flag!=1:
                sum_pval += max_rat_pval[0][1]
            else:
                sum_pval += 1

            avg_incrtness += incrt
            avg_incrtness_w_n += incrt_w_n
            if success == 1:
                count += 1
            loop += 1

        avg_pval = sum_pval/2000
        avg_incrtness = avg_incrtness/2000
        avg_incrtness_w_n = avg_incrtness_w_n/2000
        v = pd.DataFrame(data=np.array([[nop,mum,count/2000,avg_pval,avg_incrtness,avg_incrtness_w_n]]))
        v.to_csv(path+save_string+'.csv',mode = 'a',header=False,index=False)
        return {"nop":nop,"mum": mum,"success_rate":count/2000,'avg_pval':avg_pval,'avg_incrt':avg_incrtness, 'avg_incrt_w_n':avg_incrtness_w_n}

    opt_var = parallel(mumf,number_of_options,mu_m)
    csv(data = opt_var,file = path+save_string+"last.csv")
    linePlot(data_len= number_of_options,data_legend = number_of_options,array= opt_var,var= "nop", plt_var="mum",x_name=r'$\mu_m$',\
        y_name = "P-values",title="Number of options",save_name=path + save_string+"Pval.pdf",y_var=["avg_pval"])

    linePlot(data_len= number_of_options,data_legend = number_of_options + number_of_options,array= opt_var,var= "nop", plt_var="mum",x_name=r'$\mu_m$',\
        y_name = "Wrong ranking cost",title="Number of options",save_name=path + save_string+"WRC.pdf",y_var=["avg_incrt",'avg_incrt_w_n'])
