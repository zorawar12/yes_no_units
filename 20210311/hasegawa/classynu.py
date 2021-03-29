# Nature of collective-decision making by simple
# votes/no decision units.

# Author: Swadhin Agrawal
# E-mail: swadhin20@iiserb.ac.in

import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import pandas as pd
import os
import time
import random_number_generator as rng
from numba import  njit

path = os.getcwd() + "/results/"

class Decision_making:
    def __init__(self,number_of_options,err_type,mu_assessment_err,sigma_assessment_err):
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
        self.votes = self.counter(self.number_of_options,self.mu_assessment_err,self.sigma_assessment_err,assigned_units,self.err_type,Dx)
    
    @staticmethod
    @njit(parallel = True)
    def counter(nop,e_mu,e_sigma,ass_u,e_type,Dx):
        votes = [0 for i in range(nop)]
        for i in range(nop):
            for j in range(len(ass_u[i])):
                assesment_error = round(np.random.normal(e_mu,e_sigma),e_type)
                if (ass_u[i][j] < (Dx[i] + assesment_error)):
                    votes[i] += 1
        return votes

    def best_among_bests(self,ref_highest_quality):
        """
        Returns success/failure of decision making when there are multiple correct decisions as per the units
        """
        available_opt = np.array(np.where(np.array(self.votes) == max(self.votes)))[0]
        # print([max(self.votes)])
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

class workFlow:
    def __init__(self):
        self.check_count = 0
    
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
        
    def one_run(self,number_of_options=None,mu_m_1=None,sigma_m_1=None,mu_m_2=None,sigma_m_2=None,h_type=3,\
        mu_h_1=None,sigma_h_1=None,mu_h_2=None,sigma_h_2=None,x_type=3,mu_x_1=None,sigma_x_1=None,mu_x_2=None,sigma_x_2=None,err_type=0,mu_assessment_err= 0,sigma_assessment_err=0,quorum= None):

        pc = rng.units(number_of_options=number_of_options,mu=[mu_m_1,mu_m_2],sigma=[sigma_m_1,sigma_m_2])

        units_distribution = []
        
        for i in pc:
            units_distribution.append(rng.threshold(m_units=int(i),mu_h=[mu_h_1,mu_h_2],sigma_h=[sigma_h_1,sigma_h_2]))
        
        ref,qc = rng.quality(number_of_options=number_of_options,x_type=x_type,mu_x=[mu_x_1,mu_x_2],sigma_x=[sigma_x_1,sigma_x_2])

        dec = self.majority_decision(number_of_options=number_of_options,Dx = qc,\
            assigned_units=units_distribution,err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err,ref_highest_quality=ref,quorum=quorum)

        if dec == 1:
            print("success")

        else:
            print("failed")
    
    def multi_run(self,number_of_options=None,mu_m_1=None,sigma_m_1=None,mu_m_2=None,sigma_m_2=None,h_type=3,\
        mu_h_1=None,sigma_h_1=None,mu_h_2=None,sigma_h_2=None,x_type=3,mu_x_1=None,sigma_x_1=None,mu_x_2=None,sigma_x_2=None,err_type=0,mu_assessment_err= 0,sigma_assessment_err=0,quorum= None):

        pc = rng.units(number_of_options=number_of_options,mu=[mu_m_1,mu_m_2],sigma=[sigma_m_1,sigma_m_2])

        units_distribution = []
        
        for i in pc:
            units_distribution.append(rng.threshold(m_units=int(i),mu_h=[mu_h_1,mu_h_2],sigma_h=[sigma_h_1,sigma_h_2]))
        
        ref,qc = rng.quality(number_of_options=number_of_options,x_type=x_type,mu_x=[mu_x_1,mu_x_2],sigma_x=[sigma_x_1,sigma_x_2])

        dec = self.majority_decision(number_of_options=number_of_options,Dx = qc,\
            assigned_units=units_distribution,err_type=err_type,mu_assessment_err= mu_assessment_err,sigma_assessment_err=sigma_assessment_err,ref_highest_quality=ref,quorum=quorum)

        return dec
    
    @staticmethod
    @njit
    def gaussian(x,mu,sigma):
        f = 0
        for i in range(len(mu)):
            k = 1/np.sqrt(2*np.pi*sigma[i])
            f += k*np.exp(-((x-mu[i])**2)/(2*sigma[i]**2))
        return f

    @staticmethod
    @njit
    def finding_gaussian_base(area,mu,sigma,start,stop,step,x,pdf):
        if area<=0.25:
            dummy_area =0.25
            x_ = mu[0]
        elif area>0.25 and area<=0.5:
            dummy_area =0.5
            x_ = (mu[0]+mu[1])/2
        elif area>0.5 and area<=0.75:
            dummy_area =0.75
            x_ = mu[1]
        elif area>0.75 and area<=1:
            dummy_area =1
            x_ = stop
        count = np.where(x==round(x_,6))[0][0]
        while abs(dummy_area-area)>0.000001:
            dummy_area -= pdf[count]*step
            x_ -= step
            count -= 1
        return x_

class Visualization:
    def data_visualize(self,file_name,save_plot,x_var_,y_var_,z_var_,plot_type,cbar_orien=None,line_labels=None,sigma_x_1=None,\
        data =None,num_of_opts=None,delta_mu=None,sigma_x_2 = None):
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
                # t1 = time.time()
                mu = [i,i+delta_mu]
                sigma = [sigma_x_1,sigma_x_2]
                start = mu[0] - sigma[0]-5
                stop = mu[-1] + sigma[1]+5
                step = 0.0001
                dis_x = np.round(np.arange(start,stop,step),decimals=6)
                pdf =  [wf.gaussian(i,mu,sigma) for i in dis_x]
                pdf = np.multiply(pdf,1/(np.sum(pdf)*step))
                
                _1 = wf.finding_gaussian_base(1.0-(1.0/line_labels[0]),mu,sigma,start,stop,step,dis_x,pdf)
                _2 = wf.finding_gaussian_base(1.0-(1.0/line_labels[1]),mu,sigma,start,stop,step,dis_x,pdf)
                x_1.append(_1)
                x_2.append(_2)
                # t2 = time.time()
                # print(t2-t1)

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
        # plt.show()

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
        cbar. set_ticks(np.arange(0, 1, 0.04))
        ax.set_aspect('equal', 'box')
        ax.xaxis.set_ticks(np.arange(0, 10,1))
        ax.yaxis.set_ticks(np.arange(0, 10,1))
        plt.xlim(0,10)
        plt.ylim(0,10)
        plt.xlabel(x_name)
        plt.ylabel(y_name + ' and options quality')
        plt.legend()
        plt.title(title)
        plt.grid(b=True, which='major', color='black', linestyle='-',linewidth = 0.3,alpha=0.1)
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='black', linestyle='-',linewidth = 0.2,alpha=0.1)
        def onclick(event):
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %('double' if event.dblclick else 'single', event.button,event.x, event.y, event.xdata, event.ydata))
            plt.plot([0,event.xdata],[event.ydata,event.ydata],color= 'red',linewidth = 0.5)
            plt.plot([event.xdata,event.xdata],[0,event.ydata],color= 'red',linewidth = 0.5)
            ax.text(event.xdata+1, event.ydata, "(%2.2f,%2.2f)"%(event.xdata,event.ydata),fontsize=6)
            plt.savefig(save_name,format = "pdf")

        point = fig.canvas.mpl_connect('button_press_event', onclick)
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
        # plt.show()


