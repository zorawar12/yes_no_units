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
population_size = 5000
number_of_options = 10
mu_x = 0.0
sigma_x = 1.0
mu_h = 0
sigma_h = 1.0
mu_m = 100
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
sig_h = [0.0+i*0.01 for i in range(101)]
opts = [2,10]#2*i for i in range(2,6,3)]

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

    for k in range(10000):
        tc = threshold(population_size=population_size,h_type=h_type,mu_h=mu_h,sigma_h=j)
        qc = quality(number_of_options=op,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,\
            Dm = pc.Dm,Dh = tc.Dh)
        success = majority_decision(number_of_options=op,Dx = qc.Dx,\
            assigned_units= qc.assigned_units,err_type=0,mu_assessment_err= mu_assessment_err,\
            sigma_assessment_err=sigma_assessment_err,ref_highest_quality=qc.ref_highest_quality)
        if success == 1:
            count += 1
    opt_var.append({"opt":op,"sigma": j, "success_rate":count/10000})
    return opt_var

with Pool(20) as p:
    opt_var = p.starmap(sighf,inp)

options = [[] for i in range(len(opts))]
for i in opt_var:
    for j in i:
        options[opts.index(j["opt"])].append(j)
opt_var = options
#%%
c = ["blue","green","red","purple","brown"]
count = 0
fig = plt.figure()
for i in opt_var:
    plt.scatter(list(map(itemgetter("sigma"), i)),list(map(itemgetter("success_rate"), i)),c = c[count],s=0.3)    
    count += 1
plt.xlabel('Sigma_h')
plt.ylabel('Rate_of_correct_choice')
plt.legend(opts,markerscale = 10, title = "Number of options")
plt.show()

#%%
# Majority based decision with varying number of options and mu_h 
m_h = [-4.0+i*0.08 for i in range(101)]
opts = [2,10]#2*i for i in range(2,6,3)]

inp = []
for i in opts:
    for j in m_h:
        inp.append((i,j))

opt_var = []
pc = None

def muhf(op,j):
    global opt_var, pc
    count = 0
    pc = units(population_size=population_size,number_of_options=op,\
            mu_m=mu_m,sigma_m=sigma_m)

    for k in range(5000):
        tc = threshold(population_size=population_size,h_type=h_type,mu_h=j,sigma_h=sigma_h)
        qc = quality(number_of_options=op,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,\
            Dm = pc.Dm,Dh = tc.Dh)
        success = majority_decision(number_of_options=op,Dx = qc.Dx,\
            assigned_units= qc.assigned_units,err_type=0,mu_assessment_err= mu_assessment_err,\
            sigma_assessment_err=sigma_assessment_err,ref_highest_quality=qc.ref_highest_quality)
        if success == 1:
            count += 1
    opt_var.append({"opt":op,"mu": j, "success_rate":count/5000})
    return opt_var

with Pool(20) as p:
    opt_var = p.starmap(muhf,inp)

options = [[] for i in range(len(opts))]
for i in opt_var:
    for j in i:
        options[opts.index(j["opt"])].append(j)
opt_var = options
#%%
c = ["blue","green","red","purple","brown"]
count = 0
fig = plt.figure()
for i in opt_var:
    plt.scatter(list(map(itemgetter("mu"), i)),list(map(itemgetter("success_rate"), i)),c = c[count],s=0.3)    
    count += 1
plt.xlabel('Mu_h')
plt.ylabel('Rate_of_correct_choice')
plt.legend(opts,markerscale = 10, title = "Number of options")
plt.show()

#%%
# Majority based decision with varying number of options and mu_h 
number_of_options = [i for i in range(1,51,1)]
mu_h = [0,2]

inp = []
for i in mu_h:
    for j in number_of_options:
        inp.append((i,j))

mu_h_var = []
pc = None

def nf(muh,nop):
    global mu_h_var, pc
    count = 0
    pc = units(population_size=population_size,number_of_options=nop,mu_m=mu_m,sigma_m=sigma_m)
    for k in range(3000):
        tc = threshold(population_size=population_size,h_type=h_type,mu_h=muh,sigma_h=sigma_h)
        qc = quality(number_of_options=nop,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,Dm = pc.Dm,Dh = tc.Dh)
        success = majority_decision(number_of_options=nop,Dx = qc.Dx,\
            assigned_units= qc.assigned_units,err_type=0,mu_assessment_err= mu_assessment_err,\
            sigma_assessment_err=sigma_assessment_err,ref_highest_quality=qc.ref_highest_quality)
        if success == 1:
            count += 1
    mu_h_var.append({"nop":nop,"muh": muh, "success_rate":count/3000})
    return mu_h_var

with Pool(20) as p:
    mu_h_var = p.starmap(nf,inp)


mean_h = [[] for i in range(len(mu_h))]
for i in mu_h_var:
    for j in i:
        mean_h[mu_h.index(j["muh"])].append(j)
mu_h_var = mean_h
#%%
c = ["blue","green","red","purple","brown"]
count = 0
fig = plt.figure()
for i in mu_h_var:
    plt.scatter(list(map(itemgetter("nop"), i)),list(map(itemgetter("success_rate"), i)),c = c[count],s=0.3)    
    count += 1
plt.xlabel('number_of_options')
plt.ylabel('Rate_of_correct_choice')
plt.legend(mu_h,markerscale = 10, title = "mu_h")
plt.show()



#%%
# Majority based decision with varying number of options and mu_h 
m_h = [-4.0+i*0.08 for i in range(101)]
opts = [2,10]#2*i for i in range(2,6,3)]

inp = []
for i in opts:
    for j in m_h:
        inp.append((i,j))

opt_var = []
pc = None

def muhf(op,j):
    global opt_var, pc
    count = 0
    pc = units(population_size=population_size,number_of_options=op,\
            mu_m=mu_m,sigma_m=sigma_m)

    for k in range(5000):
        tc = threshold(population_size=population_size,h_type=h_type,mu_h=j,sigma_h=sigma_h)
        qc = quality(number_of_options=op,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,\
            Dm = pc.Dm,Dh = tc.Dh)
        success = majority_decision(number_of_options=op,Dx = qc.Dx,\
            assigned_units= qc.assigned_units,err_type=0,mu_assessment_err= mu_assessment_err,\
            sigma_assessment_err=sigma_assessment_err,ref_highest_quality=qc.ref_highest_quality)
        if success == 1:
            count += 1
    opt_var.append({"opt":op,"mu": j, "success_rate":count/5000})
    return opt_var

with Pool(20) as p:
    opt_var = p.starmap(muhf,inp)

options = [[] for i in range(len(opts))]
for i in opt_var:
    for j in i:
        options[opts.index(j["opt"])].append(j)
opt_var = options
#%%
c = ["blue","green","red","purple","brown"]
count = 0
fig = plt.figure()
for i in opt_var:
    plt.scatter(list(map(itemgetter("mu"), i)),list(map(itemgetter("success_rate"), i)),c = c[count],s=0.3)    
    count += 1
plt.xlabel('Mu_h')
plt.ylabel('Rate_of_correct_choice')
plt.legend(opts,markerscale = 10, title = "Number of options")
plt.show()

#%%
# Majority based decision with varying number of options and mu_h 
mu_m = [i for i in range(1,101,1)]
number_of_options = [2,10]

inp = []
for i in number_of_options:
    for j in mu_m:
        inp.append((i,j))

nop_var = []
pc = None

def mumf(nop,mum):
    global nop_var, pc
    count = 0
    pc = units(population_size=population_size,number_of_options=nop,mu_m=mum,sigma_m=sigma_m)
    for k in range(3000):
        tc = threshold(population_size=population_size,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h)
        qc = quality(number_of_options=nop,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,Dm = pc.Dm,Dh = tc.Dh)
        success = majority_decision(number_of_options=nop,Dx = qc.Dx,\
            assigned_units= qc.assigned_units,err_type=0,mu_assessment_err= mu_assessment_err,\
            sigma_assessment_err=sigma_assessment_err,ref_highest_quality=qc.ref_highest_quality)
        if success == 1:
            count += 1
    nop_var.append({"nop":nop,"mum": mum, "success_rate":count/3000})
    return nop_var

with Pool(20) as p:
    nop_var = p.starmap(mumf,inp)
#%%

mean_m = [[] for i in range(len(number_of_options))]
for i in nop_var:
    for j in i:
        mean_m[number_of_options.index(j["nop"])].append(j)
nop_var = mean_m
#%%

c = ["blue","green","red","purple","brown"]
count = 0
fig = plt.figure()
for i in nop_var:
    plt.scatter(list(map(itemgetter("mum"), i)),list(map(itemgetter("success_rate"), i)),c = c[count],s=0.3)    
    count += 1
plt.xlabel('number_of_units(variance = 0)')
plt.ylabel('Rate_of_correct_choice')
plt.legend(number_of_options,markerscale = 10, title = "Number_of_options")
plt.show()


#%%
# Majority based decision with varying number of options and mu_h 
sigma_m = [1+i*0.03 for i in range(0,1000,1)]
number_of_options = [2,10]

inp = []
for i in number_of_options:
    for j in sigma_m:
        inp.append((i,j))

nop_var = []
pc = None

def sigmf(nop,sigm):
    global nop_var, pc
    count = 0
    for k in range(10000):
        pc = units(population_size=population_size,number_of_options=nop,mu_m=mu_m,sigma_m=sigm)
        tc = threshold(population_size=population_size,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h)
        qc = quality(number_of_options=nop,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,Dm = pc.Dm,Dh = tc.Dh)
        success = majority_decision(number_of_options=nop,Dx = qc.Dx,\
            assigned_units= qc.assigned_units,err_type=0,mu_assessment_err= mu_assessment_err,\
            sigma_assessment_err=sigma_assessment_err,ref_highest_quality=qc.ref_highest_quality)
        if success == 1:
            count += 1
    nop_var.append({"nop":nop,"sigm": sigm, "success_rate":count/10000})
    return nop_var

with Pool(20) as p:
    nop_var = p.starmap(sigmf,inp)
#%%

mean_m = [[] for i in range(len(number_of_options))]
for i in nop_var:
    for j in i:
        mean_m[number_of_options.index(j["nop"])].append(j)
nop_var = mean_m
#%%

c = ["blue","green","red","purple","brown"]
count = 0
fig = plt.figure()
for i in nop_var:
    plt.scatter(list(map(itemgetter("sigm"), i)),list(map(itemgetter("success_rate"), i)),c = c[count],s=0.3)    
    count += 1
plt.xlabel('number_of_units(variance = 0)')
plt.ylabel('Rate_of_correct_choice')
plt.legend(number_of_options,markerscale = 10, title = "Number_of_options")
plt.show()
plt.savefig('sigma_m_vs_rate_of_correct_choice.pdf')

# %%
