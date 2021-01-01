# Nature of collective-decision making by simple yes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in
import csv
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

wrong_ranking_cost_graphics = 1
not_considering_no_in_ranking = 0
not_considering_no_in_ranking_sigma = 0

def units(mu_m,sigma_m,number_of_options):
    a = np.array(np.round(np.random.normal(mu_m,sigma_m,number_of_options),decimals=0))
    while a.any()<=0.:
        a = np.array(np.round(np.random.normal(mu_m,sigma_m,number_of_options),decimals=0))
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
        return majority_dec,DM.yes_stats,DM.max_ratio_pvalue,incorrectness,incorrectness_w_n
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

if wrong_ranking_cost_graphics==1:
    mu_m = [i for i in range(500,2001)]
    sigma_m = [i for i in range(160,180)]
    number_of_options = 10

    def mumf(sigm,mum):
        count = 0
        sum_pval = 0
        avg_incrtness = 0
        avg_incrtness_w_n = 0
        avg_correct_ranking = 0
        for k in range(1000):
            success ,yes_test,max_rat_pval,incrt,incrt_w_n = main_process_flow(mu_m=mum,sigma_m=sigm,err_type=0)
            sum_pval += max_rat_pval[1]
            avg_incrtness += incrt
            avg_incrtness_w_n += incrt_w_n
            if incrt_w_n == 0:
                avg_correct_ranking +=1
            if success == 1:
                count += 1
        avg_pval = sum_pval/1000
        avg_incrtness = avg_incrtness/1000
        avg_incrtness_w_n = avg_incrtness_w_n/1000
        avg_correct_ranking = avg_correct_ranking/1000
        return {"sigm":sigm,"mum": mum, "success_rate":count/1000,'avg_pval':avg_pval,'avg_incrt':avg_incrtness, 'avg_incrt_w_n':avg_incrtness_w_n,'avg_correct_ranking':avg_correct_ranking}

    opt_var = parallel(mumf,sigma_m,mu_m)
    csv(opt_var,path+'WRC_graphics_4.csv')
#    op = pd.read_csv(path+'WRC_graphics_3.csv')
#    opt_var = []
#
#    for j in range(len(op['mum'])):
#        a = {}
#        for i in op:
#            a[str(i)] = op[str(i)][j]
#        opt_var.append(a)

    graphicPlot(a=mu_m,b=sigma_m,zvar="avg_incrt",array=opt_var,x_name=r"$\sigma_m$",y_name=r"$\mu_m$",title='number of options = '+str(number_of_options),save_name=path+"wrc_graphics_4.pdf",bar_label='Wrong ranking cost')
    graphicPlot(a=mu_m,b=sigma_m,zvar="avg_correct_ranking",array=opt_var,x_name = r"$\sigma_m$",y_name=r"$\mu_m$",title="number of options = "+str(number_of_options),save_name=path+"RCR_4.pdf",bar_label='Rate of correct ranking')
    graphicPlot(a=mu_m,b=sigma_m,zvar="avg_incrt_w_n",array=opt_var,x_name = r"$\sigma_m$",y_name=r"$\mu_m$",title="number of options = "+str(number_of_options),save_name=path+"wrc_mu_sigma_m_4.pdf",bar_label='Wrong ranking cost')


if not_considering_no_in_ranking ==1:
    mu_m = [i for i in range(500,2000,100)]
    number_of_options = [2,5,10,20]

    def mumf(nop,mum):
        count = 0
        sum_pval = 0
        avg_incrtness = 0
        avg_incrtness_w_n = 0
        for k in range(2000):
            success ,yes_test,max_rat_pval,incrt,incrt_w_n = main_process_flow(mu_m=mum,number_of_options=nop,err_type=0)
            sum_pval += max_rat_pval[1]
            avg_incrtness += incrt
            avg_incrtness_w_n += incrt_w_n
            if success == 1:
                count += 1
        avg_pval = sum_pval/2000
        avg_incrtness = avg_incrtness/2000
        avg_incrtness_w_n = avg_incrtness_w_n/2000
        return {"nop":nop,"mum": mum,"success_rate":count/2000,'avg_pval':avg_pval,'avg_incrt':avg_incrtness, 'avg_incrt_w_n':avg_incrtness_w_n}

    opt_var = parallel(mumf,number_of_options,mu_m)

    plt_show(data_len= number_of_options,data_legend = number_of_options,array= opt_var,var= "nop", plt_var="mum",x_name='mean_number_of_units(variance = 150)',\
        y_name = "P-values",title="Number_of_options",save_name="number_of_units_vs_pvalue.pdf",y_var=["avg_pval"])

    plt_show(data_len= number_of_options,data_legend = number_of_options + number_of_options,array= opt_var,var= "nop", plt_var="mum",x_name='mean_number_of_units(variance = 150)',\
        y_name = "Measure_of_incorrectness",title="Number_of_options",save_name="number_of_units_vs_measure_of_incorrectness.pdf",y_var=["avg_incrt",'avg_incrt_w_n'])

if not_considering_no_in_ranking_sigma ==1:
    mu_m = [i for i in range(500,2000,100)]
    number_of_options = [2,5,10,20]
    sigma_m = [20,150]

    def musigmamf(sigmam,nop,mum):
        count = 0
        sum_pval = 0
        avg_incrtness = 0
        avg_incrtness_w_n = 0
        for k in range(2000):
            success ,yes_test,max_rat_pval,incrt,incrt_w_n = main_process_flow(mu_m=mum,number_of_options=nop,sigma_m=sigmam,err_type=0)
            sum_pval += max_rat_pval[1]
            avg_incrtness += incrt
            avg_incrtness_w_n += incrt_w_n
            if success == 1:
                count += 1
        avg_pval = sum_pval/2000
        avg_incrtness = avg_incrtness/2000
        avg_incrtness_w_n = avg_incrtness_w_n/2000
        return {"nop":nop,"mum": mum,'sigmam':sigmam,"success_rate":count/2000,'avg_pval':avg_pval,'avg_incrt':avg_incrtness, 'avg_incrt_w_n':avg_incrtness_w_n}

    opt_var = parallel3(musigmamf,sigma_m,number_of_options,mu_m)
    csv(opt_var,'not_considering_no_in_ranking_sigma.csv')
    plt_show(data_len= sigma_m,data_legend = number_of_options + number_of_options,array=opt_var,var= "sigmam", plt_var="mum",x_name=r'$\mu_m$',\
        y_name = "Wrong ranking cost",title="Number_of_options",save_name="number_of_units_vs_measure_of_incorrectness.pdf",y_var=["avg_incrt_w_n"])
