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
number_of_options = 20
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

def units(mu_m,sigma_m,number_of_options):
    """
    Arguments:
    mu_m -(int) mean of distribution from where to choose number of units to be assigned
    sigma_m - (int) standard deviation of distribution from where to choose number of units to be assigned
    Returns:
    Number of units to be assigned to an option choosen from N(mu_m,sigma_m) (array[1xn])
    """
    return np.round(np.random.normal(mu_m,sigma_m,number_of_options),decimals=0)

def threshold(m_units,h_type,mu_h,sigma_h):
    """
    Creates threshold distribution
    Arguments:
    m_units - (int from [1xn])number of assigned units to an option
    h_type - (int) number of decimal places of thresholds
    mu_h - (float) mean of distribution to select threshold for each unit from
    sigma_h - (float) standard deviation of distribution to select threshold for each unit from

    Returns: 
    Array[1xm_units] of thesholds for each assigned set of units to options
    """
    return np.round(np.random.normal(mu_h,sigma_h,m_units),decimals=h_type)

def quality(number_of_options,x_type,mu_x,sigma_x):
    """
    Creates quality stimulus
    Arguments:
    number_of_options - (int)number of option for which quality stimulus has to be assigned
    x_type - (int) number of decimal places of quality stimulus
    mu_x - (float) mean of distribution to select quality stimulus for each option from
    sigma_x - (float) standard deviation of distribution to select quality stimulus for each option from
    Returns: 
    Array[1 x number_of_options] of qualities for each option
    """
    QC = yn.qualityControl(number_of_options=number_of_options,x_type=x_type)
    QC.mu_x = mu_x
    QC.sigma_x = sigma_x
    QC.dx()
    QC.ref_highest_qual()
    return QC

def majority_decision(number_of_options,Dx,assigned_units,err_type,\
    mu_assessment_err,sigma_assessment_err,ref_highest_quality,\
        one_correct_opt = 1):
    """
    Majority based decision

    Arguments:
    number_of_options - (int)number of options to choose best from
    Dx - ([1 x number_of_options]) array of quality stimulus for each options (options sorted from highest quality to lowest quality)
    assigned_units - ([[1 x m_units] for i in range(len(number_of_options)]) array of assigned units with thresholds to each options
    err_type - (int) number of decimal places of noise
    mu_assessment_err - (float) mean of distribution to choose noise from
    sigma_assessment_err - (float) standard deviation of distribution to choose noise from
    ref_highest_quality - highest quality option to measure success
    one_correct_opt - (1,0) '0' means many correct options
    Returns:    ,
    success(1) or failure(0)

    """
    DM = yn.Decision_making(number_of_options=number_of_options,err_type=err_type,\
    mu_assessment_err=mu_assessment_err,sigma_assessment_err=sigma_assessment_err)

    DM.vote_counter(assigned_units,Dx)

    plt.scatter(Dx,DM.votes)
    plt.show()

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
    

def one_run(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h,\
    x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err):

    pc = np.array(units(number_of_options=number_of_options,mu_m=mu_m,sigma_m=sigma_m)).astype(int)

    units_distribution = []
    for i in pc:
        units_distribution.append(threshold(m_units = i ,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h))

    qc = quality(number_of_options=number_of_options,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x)

    dec = majority_decision(number_of_options=number_of_options,Dx = qc.Dx,assigned_units= units_distribution,\
        err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err,\
        ref_highest_quality=qc.ref_highest_quality)                                   

    if dec == 1:
        print("success")
    else:
        print("failed")


#%%
# Without assesment error Majority based decision
one_run()

#%%
# With assessment error Majority based decision
one_run(sigma_assessment_err=0.1)

#%%
# Random choice when more than one option's correct Majority based decision
one_run(x_type=0,err_type=0)

#%%
# Majority based Rate of correct choice as a function of sigma_h for varying number of options
sig_h = [0.0+i*0.01 for i in range(101)]
opts = [2,10]#2*i for i in range(2,6,3)]

inp = []
for i in opts:
    for j in sig_h:
        inp.append((i,j))

opt_var = []
pc = None

def sighf(op,j):
    global pc
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
    opt_va = {"opt":op,"sigma": j, "success_rate":count/2000}
    return opt_va

with Pool(8) as p:
    opt_var = p.starmap(sighf,inp)

c = ["blue","green","red","purple","brown"]
count = 0
fig = plt.figure()
data = [[] for i in range(len(opts))]

for i in opt_var:
    data[opts.index(i["opt"])].append(i)

for i in data:
    plt.scatter(list(map(itemgetter("sigma"), i)),list(map(itemgetter("success_rate"), i)),c = c[count],s=0.3)    
    count += 1

plt.xlabel('Sigma_h')
plt.ylabel('Rate_of_correct_choice')
plt.legend(opts,markerscale = 10, title = "Number of options")
plt.savefig("Sigma_h_vs_Rate_of_correct_choice.pdf",format = "pdf")
plt.show()

#%%
# Majority based Rate of correct choice as a function of mu_h for varying number of options 
m_h = [-4.0+i*0.08 for i in range(101)]
opts = [2,10]#2*i for i in range(2,6,3)]

inp = []
for i in opts:
    for j in m_h:
        inp.append((i,j))

opt_var = []
pc = None

def muhf(op,j):
    global pc
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
    opt_va = {"opt":op,"mu": j, "success_rate":count/2000}
    return opt_va

with Pool(8) as p:
    opt_var = p.starmap(muhf,inp)

c = ["blue","green","red","purple","brown"]
count = 0
fig = plt.figure()
data = [[] for i in range(len(opts))]

for i in opt_var:
    data[opts.index(i["opt"])].append(i)

for i in data:
    plt.scatter(list(map(itemgetter("mu"), i)),list(map(itemgetter("success_rate"), i)),c = c[count],s=0.3)
    count += 1

plt.xlabel('Mu_h')
plt.ylabel('Rate_of_correct_choice')
plt.legend(opts,markerscale = 10, title = "Number of options")
plt.savefig("Mu_h_vs_Rate_of_correct_choice.pdf",format = "pdf")
plt.show()

#%%
# Majority based Rate of correct choice as a function of number of options for varying mu_h 
number_of_options = [i for i in range(1,51,1)]
mu_h = [0,2]

inp = []
for i in mu_h:
    for j in number_of_options:
        inp.append((i,j))

mu_h_var = []
pc = None

def nf(muh,nop):
    global pc
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
    mu_h_va = {"nop":nop,"muh": muh, "success_rate":count/3000}
    return mu_h_va

with Pool(8) as p:
    mu_h_var = p.starmap(nf,inp)

c = ["blue","green","red","purple","brown"]
count = 0
fig = plt.figure()
data = [[] for i in range(len(mu_h))]

for i in mu_h_var:
    data[mu_h.index(i["muh"])].append(i)

for i in data:
    plt.scatter(list(map(itemgetter("nop"), i)),list(map(itemgetter("success_rate"), i)),c = c[count],s=0.3)
    count += 1

plt.xlabel('number_of_options')
plt.ylabel('Rate_of_correct_choice')
plt.legend(mu_h,markerscale = 10, title = "mu_h")
plt.savefig("number_of_options_vs_Rate_of_correct_choice.pdf",format = "pdf")
plt.show()

#%%
# Majority based Rate of correct choice as a function of mu_m for varying number of options 
mu_m = [i for i in range(1,101,1)]
number_of_options = [2,10]

inp = []
for i in number_of_options:
    for j in mu_m:
        inp.append((i,j))

nop_var = []
pc = None

def mumf(nop,mum):
    global pc
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
    nop_va = {"nop":nop,"mum": mum, "success_rate":count/3000}
    return nop_va

with Pool(8) as p:
    nop_var = p.starmap(mumf,inp)

c = ["blue","green","red","purple","brown"]
count = 0
fig = plt.figure()
data = [[] for i in range(len(number_of_options))]

for i in nop_var:
    data[number_of_options.index(i["nop"])].append(i)

for i in data:
    plt.scatter(list(map(itemgetter("mum"), i)),list(map(itemgetter("success_rate"), i)),c = c[count],s=0.3)    
    count += 1

plt.xlabel('number_of_units(variance = 0)')
plt.ylabel('Rate_of_correct_choice')
plt.legend(number_of_options,markerscale = 10, title = "Number_of_options")
plt.savefig("number_of_units_vs_Rate_of_correct_choice.pdf",format = "pdf")
plt.show()

#%%
# Majority based Rate of correct choice as a function of sigma_m for varying number of options
sigma_m = [1+i*0.03 for i in range(0,1000,1)]
number_of_options = [2,10]

inp = []
for i in number_of_options:
    for j in sigma_m:
        inp.append((i,j))

nop_var = []
pc = None

def sigmf(nop,sigm):
    global pc
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
    nop_va = {"nop":nop,"sigm": sigm, "success_rate":count/10000}
    return nop_va

with Pool(8) as p:
    nop_var = p.starmap(sigmf,inp)

c = ["blue","green","red","purple","brown"]
count = 0
fig = plt.figure()
data = [[] for i in range(len(number_of_options))]

for i in nop_var:
    data[number_of_options.index(i["nop"])].append(i)

for i in data:
    plt.scatter(list(map(itemgetter("sigm"), i)),list(map(itemgetter("success_rate"), i)),c = c[count],s=0.3)    
    count += 1

plt.xlabel('number_of_units(variance = 0)')
plt.ylabel('Rate_of_correct_choice')
plt.legend(number_of_options,markerscale = 10, title = "Number_of_options")
plt.savefig('sigma_m_vs_rate_of_correct_choice.pdf')
plt.show()

#%%
# Majority based Rate of correct choice as a function of sigma_h for varying mu_h
sig_h = [np.round(0.0+i*0.01,decimals=2) for i in range(101)]
mu_h = [np.round(-4.0+i*0.08,decimals=2) for i in range(101)]
opts = 10

inp = []
for i in mu_h:
    for j in sig_h:
        inp.append((i,j))

mu_var = []
pc = None

def sighf(mu,sig):
    global pc
    count = 0
    pc = units(population_size=population_size,number_of_options=opts,\
            mu_m=mu_m,sigma_m=sigma_m)
    for k in range(2000):
        tc = threshold(population_size=population_size,h_type=h_type,mu_h=mu,sigma_h=sig)
        qc = quality(number_of_options=opts,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,\
            Dm = pc.Dm,Dh = tc.Dh)
        success = majority_decision(number_of_options=opts,Dx = qc.Dx,\
            assigned_units= qc.assigned_units,err_type=0,mu_assessment_err= mu_assessment_err,\
            sigma_assessment_err=sigma_assessment_err,ref_highest_quality=qc.ref_highest_quality)
        if success == 1:
            count += 1
    mu_va = {"mu":mu,"sigma": sig, "success_rate":count/2000}
    return mu_va

with Pool(8) as p:
    mu_var = p.starmap(sighf,inp)

fig, ax = plt.subplots()
z = np.array(list(map(itemgetter("success_rate"), mu_var))).reshape(len(mu_h),len(sig_h))
cs = ax.contourf(sig_h,mu_h,z)   
cbar = fig.colorbar(cs)
plt.xlabel('Sigma_h')
plt.ylabel('Mu_h')
plt.legend(title = "Number_of_options = 10")
plt.savefig("mu_h_vs_sigma_h.pdf",format = "pdf")
plt.show()
# %%
# Majority based Rate of correct choice as a function of mu_x for varying mu_h
mu_x = [np.round(-4.0+i*0.08,decimals=2) for i in range(101)]
mu_h = [np.round(-4.0+i*0.08,decimals=2) for i in range(101)]
opts = 10

inp = []
for i in mu_h:
    for j in mu_x:
        inp.append((i,j))

mu_var = []
pc = None

def sighf(muh,mux):
    global pc
    count = 0
    pc = units(population_size=population_size,number_of_options=opts,\
            mu_m=mu_m,sigma_m=sigma_m)
    for k in range(2000):
        tc = threshold(population_size=population_size,h_type=h_type,mu_h=muh,sigma_h=sigma_h)
        qc = quality(number_of_options=opts,x_type=x_type,mu_x=mux,sigma_x=sigma_x,\
            Dm = pc.Dm,Dh = tc.Dh)
        success = majority_decision(number_of_options=opts,Dx = qc.Dx,\
            assigned_units= qc.assigned_units,err_type=0,mu_assessment_err= mu_assessment_err,\
            sigma_assessment_err=sigma_assessment_err,ref_highest_quality=qc.ref_highest_quality)
        if success == 1:
            count += 1
    mu_va = {"mux":mux,"muh": muh, "success_rate":count/2000}
    return mu_va

with Pool(8) as p:
    mu_var = p.starmap(sighf,inp)

fig, ax = plt.subplots()
z = np.array(list(map(itemgetter("success_rate"), mu_var))).reshape(len(mu_h),len(mu_x))
cs = ax.contourf(mu_x,mu_h,z)
cbar = fig.colorbar(cs)
plt.xlabel('Mu_x')
plt.ylabel('Mu_h')
plt.legend(title = "Number_of_options = 10")
plt.savefig("mu_h_vs_mu_x.pdf",format = "pdf")
plt.show()

# %%
# Majority based Rate of correct choice as a function of sigma_x for varying sigma_h
sig_x = [np.round(i*0.04,decimals=2) for i in range(101)]
sig_h = [np.round(i*0.04,decimals=2) for i in range(101)]
opts = 10

inp = []
for i in sig_h:
    for j in sig_x:
        inp.append((i,j))

sig_var = []
pc = None

def sighf(sigh,sigx):
    global pc
    count = 0
    pc = units(population_size=population_size,number_of_options=opts,\
            mu_m=mu_m,sigma_m=sigma_m)
    for k in range(2000):
        tc = threshold(population_size=population_size,h_type=h_type,mu_h=mu_h,sigma_h=sigh)
        qc = quality(number_of_options=opts,x_type=x_type,mu_x=mu_x,sigma_x=sigx,\
            Dm = pc.Dm,Dh = tc.Dh)
        success = majority_decision(number_of_options=opts,Dx = qc.Dx,\
            assigned_units= qc.assigned_units,err_type=0,mu_assessment_err= mu_assessment_err,\
            sigma_assessment_err=sigma_assessment_err,ref_highest_quality=qc.ref_highest_quality)
        if success == 1:
            count += 1
    sig_va = {"sigx":sigx,"sigh": sigh, "success_rate":count/2000}
    return sig_va

with Pool(8) as p:
    sig_var = p.starmap(sighf,inp)

fig, ax = plt.subplots()
z = np.array(list(map(itemgetter("success_rate"), sig_var))).reshape(len(sig_h),len(sig_x))
cs = ax.contourf(sig_x,sig_h,z)
cbar = fig.colorbar(cs)
plt.xlabel('Sigma_x')
plt.ylabel('Sigma_h')
plt.legend(title = "Number_of_options = 10")
plt.savefig("sig_h_vs_sig_x.pdf",format = "pdf")
plt.show()

# %%
# Majority based Rate of correct choice as a function of quorum for varying sigma_m
quorum = [i for i in range(100)]
sig_m = [0,30]
opts = 10

inp = []
for i in sig_m:
    for j in quorum:
        inp.append((i,j))

sig_var = []
pc = None

def sigmf(sigm,quo):
    global pc
    count = 0
    pc = units(population_size=population_size,number_of_options=opts,\
            mu_m=mu_m,sigma_m=sigm)
    for k in range(2000):
        tc = threshold(population_size=population_size,h_type=h_type,mu_h=mu_h,sigma_h=sigma_h)
        qc = quality(number_of_options=opts,x_type=x_type,mu_x=mu_x,sigma_x=sigma_x,\
            Dm = pc.Dm,Dh = tc.Dh)
        success = majority_decision(number_of_options=opts,Dx = qc.Dx,\
            assigned_units= qc.assigned_units,err_type=0,mu_assessment_err= mu_assessment_err,\
            sigma_assessment_err=sigma_assessment_err,ref_highest_quality=qc.ref_highest_quality)
        if success == 1:
            count += 1
    sig_va = {"sigm":sigm,"quo": quo, "success_rate":count/2000}
    return sig_va

with Pool(8) as p:
    sig_var = p.starmap(sighf,inp)

# fig, ax = plt.subplots()
# z = np.array(list(map(itemgetter("success_rate"), sig_var))).reshape(len(sig_h),len(sig_x))
# cs = ax.contourf(sig_x,sig_h,z)
# cbar = fig.colorbar(cs)
# plt.xlabel('Sigma_x')
# plt.ylabel('Sigma_h')
# plt.legend(title = "Number_of_options = 10")
# plt.savefig("sig_h_vs_sig_x.pdf",format = "pdf")
# plt.show()