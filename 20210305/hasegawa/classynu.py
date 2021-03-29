# Nature of collective-decision making by simple 
# votes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

#%%

import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import pandas as pd
import os
import time


path = os.getcwd() + "/results/"

class Decision_making:
    def __init__(self,number_of_options,err_type,mu_assessment_err=0,sigma_assessment_err=0):
        self.err_type = err_type
        self.number_of_options = number_of_options
        self.quorum = None
        self.mu_assessment_err = mu_assessment_err
        self.sigma_assessment_err = sigma_assessment_err
        self.votes = None

    def vote_counter(self,assigned_units,Dx):
        """
        Each unit provides its decision and votes are counted for each options
        """
        votes = [0 for i in range(self.number_of_options)]
        for i in range(self.number_of_options):
            assesment_error = np.round(np.random.normal(self.mu_assessment_err,self.sigma_assessment_err,len(assigned_units[i])),decimals= self.err_type)
            for j in range(len(assigned_units[i])):
                if (assigned_units[i][j] < (Dx[i] + assesment_error[j])):
                    votes[i] += 1
        self.votes = votes

    def best_among_bests(self,ref_highest_quality):
        """
        Returns success/failure of decision making when there are multiple correct decisions as per the units
        """
        available_opt = np.array(np.where(np.array(self.votes) == max(self.votes)))[0]
        opt_choosen = np.random.randint(0,len(available_opt))
        if available_opt[opt_choosen] ==  ref_highest_quality:
            return 1
        else:
            return 0

    def quorum_voting(self,assigned_units,Dx,ref_highest_quality):
        """
        success(1) or failure(0) ,quorum_reached(success(1) or failure(0)),majority decision (one_correct(success(1) or failure(0)),multi_correct(success(1) or failure(0)))
        """
        units_used = [0 for i in range(self.number_of_options)]
        for i in range(self.number_of_options):
            assesment_error = np.round(np.random.normal(self.mu_assessment_err,self.sigma_assessment_err,len(assigned_units[i])),decimals= self.err_type)
            count = 0
            while count<self.quorum:
                if units_used[i] == len(assigned_units[i]):
                    break
                if (assigned_units[i][units_used[i]] < (Dx[i] + assesment_error[units_used[i]])):
                    count += 1
                units_used[i] += 1
        units_used = np.array(units_used)
        loc = np.array(np.where(units_used == min(units_used)))[0]
        opt_choosen = np.random.randint(0,len(loc))

        flag = 0
        for i in range(self.number_of_options):
            if units_used[i] == len(assigned_units[i]):
                flag += 1
        if flag==self.number_of_options:
            quorum_reached = 0
            result = 0
            return result,quorum_reached

        if loc[opt_choosen] ==  ref_highest_quality:
            result = 1
            quorum_reached = 1
            return result,quorum_reached
        else:
            quorum_reached = 1
            result = 0
            return result,quorum_reached

class qualityControl:
    def __init__(self,number_of_options,x_type):
        self.mu_x_1 = None
        self.sigma_x_1 = None
        self.mu_x_2 = None
        self.sigma_x_2 = None
        self.Dx = None
        self.x_type = x_type
        self.number_of_options = number_of_options
        self.ref_highest_quality = None

    def ref_highest_qual(self):
        """
        Provides known highest quality option
        """
        best_list = np.array(np.where(self.Dx == max(self.Dx)))[0]
        opt_choosen = np.random.randint(0,len(best_list))
        self.ref_highest_quality = best_list[opt_choosen]

    def dx(self):
        """
        Provides distribution of quality stimulus for each option upto specified decimal places
        """
        self.Dx = []
        peak_choice = np.random.randint(0,2,self.number_of_options)
        # peak_choice = np.array([1 for i in range(int(self.number_of_options/2))])
        # peak_choice = np.append(peak_choice,np.array([0 for i in range(self.number_of_options-len(peak_choice))]))
        for i in peak_choice:
            if i==0:
                self.Dx.append(np.round(np.random.normal(self.mu_x_1,self.sigma_x_1),decimals=self.x_type))
            else:
                self.Dx.append(np.round(np.random.normal(self.mu_x_2,self.sigma_x_2),decimals=self.x_type))

class workFlow:
    def __init__(self):
        self.check_count = 0

    def units(self,mu_m_1,sigma_m_1,mu_m_2,sigma_m_2,number_of_options):
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

    def threshold(self,m_units,mu_h_1,sigma_h_1,mu_h_2,sigma_h_2):
        a = []
        peak_choice = np.random.randint(0,2,m_units)
        for i in peak_choice:
            if i==0:
                a.append(np.round(np.random.normal(mu_h_1,sigma_h_1),decimals=3))
            else:
                a.append(np.round(np.random.normal(mu_h_2,sigma_h_2),decimals=3))
        return a

    def quality(self,number_of_options,mu_x_1,sigma_x_1,mu_x_2,sigma_x_2,x_type=3):
        QC = qualityControl(number_of_options=number_of_options,x_type=x_type)
        QC.mu_x_1 = mu_x_1
        QC.sigma_x_1 = sigma_x_1
        QC.mu_x_2 = mu_x_2
        QC.sigma_x_2 = sigma_x_2
        QC.dx()
        QC.ref_highest_qual()
        return QC
    
    def majority_decision(self,number_of_options,Dx,assigned_units,ref_highest_quality,err_type=0,\
        mu_assessment_err=0,sigma_assessment_err=0,\
            quorum = None):
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
        quorum - (int) quorum to be reached first in order to consider an option as best
        Returns:
        If quorum = None then success(1) or failure(0)
        else success(1) or failure(0) ,quorum_reached(success(1) or failure(0)),majority decision (one_correct(success(1) or failure(0)),multi_correct(success(1) or failure(0)))
        """
        DM = Decision_making(number_of_options=number_of_options,err_type=err_type,\
        mu_assessment_err=mu_assessment_err,sigma_assessment_err=sigma_assessment_err)
        DM.quorum = quorum
        # t1 = time.time()
        DM.vote_counter(assigned_units,Dx)
        # t2 = time.time()
        # print(t2-t1)
        majority_dec = DM.best_among_bests(ref_highest_quality)
        if quorum == None:
            # plt.scatter(Dx,DM.votes)
            # plt.show()

            return majority_dec

        else:
            result,quorum_reached = DM.quorum_voting(assigned_units,Dx,ref_highest_quality)
            return result,quorum_reached,majority_dec
        
    def one_run(self,number_of_options,mu_m,sigma_m,mu_h,sigma_h,\
        mu_x,sigma_x,x_type=3,h_type=3,err_type=0,mu_assessment_err=0,sigma_assessment_err=0,quorum=None):

        pc = self.units(number_of_options=number_of_options,mu_m_1=mu_m,sigma_m_1=sigma_m,mu_m_2=mu_m,sigma_m_2=sigma_m)

        units_distribution = []
        
        for i in pc:
            units_distribution.append(self.threshold(m_units=i,mu_h_1=mu_h,sigma_h_1=sigma_h,mu_h_2=mu_h,sigma_h_2=sigma_h))
        
        qc = self.quality(number_of_options=number_of_options,mu_x_1=mu_x,sigma_x_1=sigma_x,mu_x_2=mu_x,sigma_x_2=sigma_x)
        
        if self.check_count==0:
            print("Number of options %f"%len(qc.Dx))
            print("Number of options %f"%len(pc))
            print("Number of options %f"%len(units_distribution))
            print("Number of units in each options %f"%len(units_distribution[0]))
            self.check_count = 1

        dec = self.majority_decision(number_of_options=number_of_options,Dx = qc.Dx,assigned_units= units_distribution,\
            err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err,\
            ref_highest_quality=qc.ref_highest_quality,quorum=quorum)

        if dec == 1:
            print("success")

        else:
            print("failed")
    
    def multi_run(self,number_of_options=None,mu_m_1=None,sigma_m_1=None,mu_m_2=None,sigma_m_2=None,h_type=3,\
        mu_h_1=None,sigma_h_1=None,mu_h_2=None,sigma_h_2=None,x_type=3,mu_x_1=None,sigma_x_1=None,mu_x_2=None,sigma_x_2=None,err_type=0,mu_assessment_err= 0,sigma_assessment_err=0,quorum= None):

        pc = self.units(number_of_options=number_of_options,mu_m_1=mu_m_1,sigma_m_1=sigma_m_1,mu_m_2=mu_m_2,sigma_m_2=sigma_m_2)

        units_distribution = []
        
        for i in pc:
            units_distribution.append(self.threshold(m_units=i,mu_h_1=mu_h_1,sigma_h_1=sigma_h_1,mu_h_2=mu_h_2,sigma_h_2=sigma_h_2))
        
        qc = self.quality(number_of_options=number_of_options,x_type=x_type,mu_x_1=mu_x_1,sigma_x_1=sigma_x_1,mu_x_2=mu_x_2,sigma_x_2=sigma_x_2)
        
        if self.check_count==0:
            print("Number of options %f"%len(qc.Dx))
            print("Number of options %f"%len(pc))
            print("Number of options %f"%len(units_distribution))
            print("Number of units in each options %f"%len(units_distribution[0]))
            self.check_count = 1

        dec = self.majority_decision(number_of_options=number_of_options,Dx = qc.Dx,\
            assigned_units=units_distribution,err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err,ref_highest_quality=qc.ref_highest_quality,quorum=quorum)

        return dec
    
    def gaussian(self,x,mu,sigma):
        k = 1/np.sqrt(2*np.pi*sigma)
        return k*np.exp(-((x-mu)**2)/(2*sigma**2))

    def finding_gaussian_base(self,area,mu,sigma):
        step = 0.00001
        x = mu
        x_ = []
        fx_ = []
        if area<0.5:
            dummy_area = 0.5
            while dummy_area-area >0.000:
                x_.append(x)
                fx = self.gaussian(x,mu,sigma)
                fx_.append(fx)
                dummy_area -= fx*step
                x -= step
            print(dummy_area)
        elif area>0.5:
            dummy_area = 0.5
            while dummy_area-area <0.000:
                x_.append(x)
                fx = self.gaussian(x,mu,sigma)
                fx_.append(fx)
                dummy_area += fx*step
                x += step
            print(dummy_area)
        else:
            x_.append(x)
            fx = self.gaussian(x,mu,sigma)
            fx_.append(fx)
            print(area)
        return [x,x_,fx_]

class Visualization:
    def data_visualize(self,file_name,save_plot,x_var_,y_var_,z_var_,plot_type,cbar_orien=None,line_labels=None,sigma_x_1=None,data =None,num_of_opts=None):
        if data == None:
            op = pd.read_csv(path+file_name)
            opt_var = []

            for j in range(len(op[x_var_])):
                a = {}
                for i in op:
                    a[str(i)] = op[str(i)][j]
                opt_var.append(a)
        else:
            opt_var = data
        
        x = []
        y = []
        z = []  # Make sure that it is in ordered form as y variable (i.e. 1st column of data file)
        for i in opt_var:
            if i[x_var_] not in x:
                x.append(i[x_var_])
            if i[y_var_] not in y:
                y.append(i[y_var_])
            z.append(i[z_var_])

            print(np.round(len(z)/len(opt_var),decimals=2),end="\r")
        print(np.round(len(z)/len(opt_var),decimals=2))

        if plot_type == 'graphics':
            wf = workFlow()
            x_1 = []
            x_2 = []
            for i in x:
                [_1,intermediate_x,intermediate_fx] = wf.finding_gaussian_base(1-(1/line_labels[0]),i,sigma_x_1)
                [_2,intermediate_x,intermediate_fx] = wf.finding_gaussian_base(1-(1/line_labels[1]),i,sigma_x_1)
                x_1.append(_1)
                x_2.append(_2)

            self.graphicPlot(a= y,b=x,array= opt_var,x_name=r'%s'%x_var_,y_name=r'%s'%y_var_,z_name="Rate of correct choice",title="Number_of_options = "+str(num_of_opts),save_name=path+save_plot+x_var_[2:-1]+y_var_[2:-1]+'RCD.pdf',cbar_loc=cbar_orien,z_var=z,options_line=[x_1,x_2],line_labels=line_labels)

        elif plot_type == 'line':
            self.linePlot(y,x,z,x_name=y_var_,y_name=z_var_, title="Number of options",save_name=path + save_plot+".pdf")

        elif plot_type == 'bar':
            self.barPlot(quorum,opt_v[str(sig_m[i])],save_name[i],"maj")

    def linePlot(self,x,y,z,x_name,y_name,title,save_name):
        c = ["blue","green","red","purple","brown","black"]
        count = 0
        fig = plt.figure()
        plt.style.use("ggplot")

        for i in range(len(y)):
            plt.plot(x,[z[s] for s in range(i*len(x),(i+1)*len(x),1)],c = c[count],linewidth = 1)
            count += 1

        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.legend(y,markerscale = 3, title = title)
        plt.savefig(save_name,format = "pdf")
        plt.show()

    def graphicPlot(self,a,b,array,x_name,y_name,z_name,title,save_name,cbar_loc,options_line,line_labels,z_var = None):
        fig, ax = plt.subplots()
        if z_var is not None:
            z = np.array(z_var).reshape(len(a),len(b))
        else:
            z = np.array(list(map(itemgetter("success_rate"), array))).reshape(len(a),len(b))
        cs = ax.pcolormesh(b,a,z)
        colors = ["black","brown"]
        
        for j in range(len(options_line)):
            plt.plot(b,options_line[j],color = colors[j],linestyle='-.',label = str(line_labels[j]))
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
        plt.ylabel(y_name + ' and options quality')
        plt.legend()
        plt.title(title)
        plt.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='black', linestyle='-',linewidth = 0.2,alpha=0.1)
        plt.savefig(save_name,format = "pdf")
        plt.show()

    def barPlot(self,quor,opt_v,save_name,correct):
        fig, ax = plt.subplots()
        ax.bar(quor,[1 for i in range(1,101,1)],width=1,color = "white",edgecolor='black')
        ax.bar(quor,list(map(itemgetter("success_rate"), opt_v)),width=1,color = "blue",edgecolor='black')
        ax.bar(quor,list(map(itemgetter("q_not_reached"), opt_v)),bottom=list(map(itemgetter("success_rate"), opt_v)),width=1,color = "orange",edgecolor='black')
        plt.plot(quor,list(map(itemgetter(correct), opt_v)),color ="red")
        plt.xlabel('Quorum size')
        plt.ylabel('Rate of choice')
        plt.savefig(save_name,format = "pdf")
        plt.show()


# %%
