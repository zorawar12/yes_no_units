# Nature of collective-decision making by simple 
# yes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin12a@iiserb.ac.in

#%%

import numpy as np
import matplotlib.pyplot as plt
import classynu as yn
from multiprocessing import Pool
from operator import itemgetter 

#%%
population_size = 2000
number_of_options = 10
mu_x = 0.0
sigma_x = 1.0
mu_h = 0.0
sigma_h = 1.0
mu_m = 200
sigma_m = 0
mu_assessment_err = 0.0
sigma_assessment_err = 0.0
x_type = 3
h_type = 3
err_type = 0
#%%

def units(population_size,number_of_options,mu_m,sigma_m):
    """
    Creates population
    """
    PC = yn.populationControl(population_size=population_size,number_of_options=number_of_options)
    PC.mu_m = mu_m
    PC.sigma_m = sigma_m
    PC.dm()
    return PC

def threshold(population_size,h_type,mu_h,sigma_h):
    """
    Creates threshold distribution
    """
    TC = yn.thresholdControl(population_size=population_size,h_type=h_type)
    TC.mu_h = mu_h
    TC.sigma_h = sigma_h
    TC.dh()
    return TC

def quality(number_of_options,x_type,mu_x,sigma_x,Dm,Dh):
    """
    Creates quality stimulus
    """
    QC = yn.qualityControl(number_of_options=number_of_options,x_type=x_type)
    QC.mu_x = mu_x
    QC.sigma_x = sigma_x
    QC.dx()
    QC.ref_highest_qual()
    QC.assign_units_to_opts(Dm,Dh)
    return QC

def majority_decision(number_of_options,Dx,assigned_units,err_type,\
    mu_assessment_err,sigma_assessment_err,ref_highest_quality,\
        one_correct_opt = 1):
    """
    Majority based decision
    """
    DM = yn.Decision_making(number_of_options=number_of_options,err_type=err_type,\
    mu_assessment_err=mu_assessment_err,sigma_assessment_err=sigma_assessment_err)

    DM.vote_counter(assigned_units,Dx)
    DM.vote_associator(Dx)

    # plt.scatter(Dx,DM.votes)
    # plt.show()

    if one_correct_opt == 1:
        if DM.one_correct(ref_highest_quality) == 1:
            return 1
        else:
            return 0
        
    else:
        if DM.multi_correct(ref_highest_quality) == 1:
            return 1
        else:
            return 0
    

#%%
# Without assesment error Majority based decision
pc = units(population_size=population_size,number_of_options=number_of_options,\
    mu_m=mu_m,sigma_m=sigma_m)
tc = threshold(population_size=population_size,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h)
qc = quality(number_of_options=number_of_options,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,\
    Dm = pc.Dm,Dh = tc.Dh)
dec = majority_decision(number_of_options=number_of_options,Dx = qc.Dx,\
    assigned_units= qc.assigned_units,err_type=err_type,mu_assessment_err= mu_assessment_err,\
    sigma_assessment_err=sigma_assessment_err,ref_highest_quality=qc.ref_highest_quality)                                   
if dec == 1:
    print("success")
else:
    print("failed")

#%%
# With assessment error Majority based decision
pc = units(population_size=population_size,number_of_options=number_of_options,\
    mu_m=mu_m,sigma_m=sigma_m)
tc = threshold(population_size=population_size,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h)
qc = quality(number_of_options=number_of_options,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,\
    Dm = pc.Dm,Dh = tc.Dh)
dec = majority_decision(number_of_options=number_of_options,Dx = qc.Dx,\
    assigned_units= qc.assigned_units,err_type=err_type,mu_assessment_err= mu_assessment_err,\
    sigma_assessment_err=0.1,ref_highest_quality=qc.ref_highest_quality)        
if dec == 1:
    print("success")
else:
    print("failed")

#%%
# Random choice when more than one option's correct Majority based decision
pc = units(population_size=population_size,number_of_options=number_of_options,\
    mu_m=mu_m,sigma_m=sigma_m)
tc = threshold(population_size=population_size,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h)
qc = quality(number_of_options=number_of_options,x_type=0,mu_x=mu_x,sigma_x=sigma_x,\
    Dm = pc.Dm,Dh = tc.Dh)
dec = majority_decision(number_of_options=number_of_options,Dx = qc.Dx,\
    assigned_units= qc.assigned_units,err_type=0,mu_assessment_err= mu_assessment_err,\
    sigma_assessment_err=sigma_assessment_err,ref_highest_quality=qc.ref_highest_quality,one_correct_opt=0)
if dec == 1:
    print("success")
else:
    print("failed")

#%%
# Majority based decision with varying number of options and sigma_h 
sig_h = [i/1000.0 for i in range(1001)]
opts = [2*i for i in range(2,5,2)]

inp = []
for i in opts:
    for j in sig_h:
        inp.append((i,j))

opt_var = []
pc = None

def sighf(op,j):
    global opt_var, pc
    count = 0
    pc = units(population_size=population_size,number_of_options=op,\
            mu_m=mu_m,sigma_m=sigma_m)

    for k in range(2000):
        tc = threshold(population_size=population_size,h_type=h_type,mu_h=mu_h,sigma_h=j)
        qc = quality(number_of_options=op,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,\
            Dm = pc.Dm,Dh = tc.Dh)
        success = majority_decision(number_of_options=op,Dx = qc.Dx,\
            assigned_units= qc.assigned_units,err_type=0,mu_assessment_err= mu_assessment_err,\
            sigma_assessment_err=sigma_assessment_err,ref_highest_quality=qc.ref_highest_quality)
        if success == 1:
            count += 1
    opt_var.append({"opt":op,"sigma": j, "success_rate":count/2000})
    return opt_var

with Pool(20) as p:
    opt_var = p.starmap(sighf,inp)

options = [[] for i in range(len(opts))]
for i in opt_var:
    for j in i:
        options[opts.index(j["opt"])].append(j)
opt_var = options
c = ["blue","green","red","purple","brown"]
count = 0
fig = plt.figure()
for i in opt_var:
    plt.scatter(list(map(itemgetter("sigma"), i)),list(map(itemgetter("success_rate"), i)),c = c[count],s=0.3)    
    count += 1
plt.show()

#%%
# Majority based decision with varying number of options and mu_h 
m_h = [i/100.0 for i in range(-400,400,1)]
opts = [2*i for i in range(2,5,2)]

inp = []
for i in opts:
    for j in sig_h:
        inp.append((i,j))

opt_var = []
pc = None

def sighf(op,j):
    global opt_var, pc
    count = 0
    pc = units(population_size=population_size,number_of_options=op,\
            mu_m=mu_m,sigma_m=sigma_m)

    for k in range(2000):
        tc = threshold(population_size=population_size,h_type=h_type,mu_h=j,sigma_h=sigma_h)
        qc = quality(number_of_options=op,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,\
            Dm = pc.Dm,Dh = tc.Dh)
        success = majority_decision(number_of_options=op,Dx = qc.Dx,\
            assigned_units= qc.assigned_units,err_type=0,mu_assessment_err= mu_assessment_err,\
            sigma_assessment_err=sigma_assessment_err,ref_highest_quality=qc.ref_highest_quality)
        if success == 1:
            count += 1
    opt_var.append({"opt":op,"mu": j, "success_rate":count/2000})
    return opt_var

with Pool(20) as p:
    opt_var = p.starmap(sighf,inp)

options = [[] for i in range(len(opts))]
for i in opt_var:
    for j in i:
        options[opts.index(j["opt"])].append(j)
opt_var = options
c = ["blue","green","red","purple","brown"]
count = 0
fig = plt.figure()
for i in opt_var:
    plt.scatter(list(map(itemgetter("mu"), i)),list(map(itemgetter("success_rate"), i)),c = c[count],s=0.3)    
    count += 1
plt.show()

#%%


#%%
# num_of_units = [j for j in range(100,20000,500)]
# num_of_opt = [j for j in range(2,20,1)]


# for j in num_of_units:
#     for i in num_of_opt:
#         ynu = yn.Decision_making(j,i)


# %%
