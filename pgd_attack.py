from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
from models.small_cnn import *
from models.net_mnist import *
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import pandas as pd
import input_add_noise as attacks
from scipy import stats
import pymc3 as pm
import arviz as az
import statistics
import pickle
from pathlib import Path

class pgd(object):

    def __init__(self, args,device,advset,testset,attack,epsilon,samplesize,row_max,dataset,model,adv,round):
        self.args = args
        self.device = device
        self.advset = advset
        self.testset = testset
        self.attack = attack
        self.epsilon = epsilon
        self.sample_size = int(samplesize)
        self.row_max = row_max
        self.dataset = dataset
        self.model = model
        self.adv = adv
        self.round = round

    def _pgd_whitebox(self,
                      model,
                      X,
                      y,
                      epsilon,
                      num_steps,
                      step_size):
        out = model(X)
        err = (out.data.max(1)[1] != y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)
        if self.args.random:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(self.device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model(X_pgd), y)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
        print('err pgd (white-box): ', err_pgd)
        return err, err_pgd

    def _pgd_blackbox(self,
                      model_target,
                      model_source,
                      X,
                      y,
                      epsilon,
                      num_steps,
                      step_size):
        out = model_target(X)
        err = (out.data.max(1)[1] != y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)
        if self.args.random:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(self.device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

        err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
        print('err pgd black-box: ', err_pgd)
        return err, err_pgd

    def _pgd_blackbox_gen(self,
                          model_target,
                          X,
                          y,
                          epsilon,
                          num_steps,
                          step_size):
        # print("Start generating data")
        out = model_target(X)
        err = (out.data.max(1)[1] != y.data).float().sum()
        X_pgd = Variable(X.data, requires_grad=True)
        if self.args.random:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(self.device)
            X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

        for ii in range(int(num_steps)):
            # print("now in iteration: " + str(ii))
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()
            with torch.enable_grad():
                loss = nn.CrossEntropyLoss()(model_target(X_pgd), y)
            loss.backward()
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
        return X_pgd

    def eval_adv_test_whitebox(self,model, device, test_loader):
        """
        evaluate model by white-box attack
        """
        model.eval()
        robust_err_total = 0
        natural_err_total = 0

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err_natural, err_robust = self._pgd_whitebox(model, X, y,self.args.epsilon,self.args.num_steps,self.args.step_size)
            robust_err_total += err_robust
            natural_err_total += err_natural
        print('natural_err_total: ', natural_err_total)
        print('robust_err_total: ', robust_err_total)

    # generate pgd data
    def gen_data(self,model_target, test_loader):
        all_datasets = []
        print("Start generating data")
        for data, target in test_loader:
            data, target = data.to(self.device), target.to(self.device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            X_pgd = self._pgd_blackbox_gen(model_target, X, y, self.args.epsilon, self.args.num_steps,
                                           self.args.step_size)
            all_datasets.append(torch.utils.data.TensorDataset(X_pgd, y))
            # break
        final_dataset = torch.utils.data.ConcatDataset(all_datasets)
        torch.save(final_dataset, "data_attack/" + self.dataset + "_" + self.model + "_" + str(self.epsilon) + ".pt")
        print("finish saving")

    def compute_individual_divergence(self,heat_dict,model_target,X_test,y_test,row_count,act_list_adv):
        out_test, act_list_test = model_target.forward_act(X_test.unsqueeze(0))
        corr_test = out_test.data.max(1)[1] == y_test.data#originally classify correctly, discard those originally classify incorrectly? seems yes
        if corr_test:
            # print("now in row: " + str(row_count))
            act_list_len = len(act_list_test)
            for j in range(act_list_len):
                sub = torch.absolute(torch.subtract(act_list_test[j], act_list_adv[j]))  # subraction is done here
                flatten_np = sub.cpu().data.numpy().flatten()
                if not j in heat_dict:
                    heat_dict[j] = [flatten_np]
                else:
                    heat_dict[j].append(flatten_np)

    def save_layer_state(self,layer,act_df_mean,act_df_var,chosen_index):
        Path("metadata/" + str(self.round)).mkdir(parents=True, exist_ok=True)#parents parameter means any parts of the path that don't exist will be created
        with open("metadata/" + str(self.round) + "/mean_" + str(self.dataset) + "_" + \
                  str(self.model) + "_" + str(layer) + "_" + str(self.attack) + "_" + \
                  str(self.sample_size) + "_" + str(self.epsilon) + \
                  ".txt", "wb") as fp:
            pickle.dump(act_df_mean, fp)
        with open("metadata/" + str(self.round) + "/var_" + str(self.dataset) + "_" + \
                  str(self.model) + "_" + str(layer) + "_" + str(self.attack) + "_" + \
                  str(self.sample_size) + "_" + str(self.epsilon) + \
                  ".txt", "wb") as fp:
            pickle.dump(act_df_var, fp)
        with open("metadata/" + str(self.round) + "/chosenindex_" + str(self.dataset) + "_" + \
                  str(self.model) + "_" + str(layer) + "_" + str(self.attack) + "_" + \
                  str(self.sample_size) + "_" + str(self.epsilon) + \
                  ".txt", "wb") as fp:
            pickle.dump(chosen_index, fp)



        # return heat_dict #all parameters in python are passed by reference
    def layer_neuron_wise_mcmc(self,heat_dict,layer,max_neu,layer_all_info):
        act_list = heat_dict[layer]  # heapmap is a list, each element is the diff in activation value
        act_len = len(act_list[0])
        print("The of samples: " + str(len(act_list)))
        chosen_neus = act_list
        if act_len > max_neu:
            act_df_var = pd.DataFrame.from_records(act_list).var().sort_values(ascending=False)  # return dataframe
            act_df_mean = pd.DataFrame.from_records(act_list).mean().sort_values(ascending=False)  # return dataframe
            # chosen_index = np.linspace(0, act_len-1, num=max_neu,dtype=int)
            chosen_index = np.arange(0, max_neu)
            chosen_act_index = [act_df_var.index[i] for i in chosen_index]
            self.save_layer_state(layer,act_df_mean,act_df_var,chosen_act_index)
            chosen_neus = []
            for instance in act_list:
                this_instance = []
                for inx in chosen_act_index:
                    this_instance.append(instance[inx])
                chosen_neus.append(this_instance)
            print("*******number of neurons: " + str(len(chosen_neus[0])))
            act_len = max_neu
        assert len(chosen_neus[0]) <= max_neu, "too many neurons"
        with pm.Model() as model:
            data = pm.Data('data', chosen_neus)
            mu = pm.Normal('mu', 0.0, 0.1, shape=act_len)  # prior, use np.mean(act_df_mean[:max_neu].to_numpy())?
            sigma = pm.Normal('sigma', 0.1, 0.1,
                              shape=act_len)  # prior, for sigma has to be log normal? use np.mean(act_df_var[:max_neu].to_numpy())?
            observe = pm.Normal('y', mu=mu, sigma=sigma,
                                observed=data)  # So either mu or sigma has to be a random variable
            t = pm.sample(act_len, chains=4, random_seed=123, init="adapt_diag")
        layer_all_info[str(layer)] = az.summary(t)  # output dataframe


    def pre_mcmc(self,heat_dict,layer_all_info):
        max_neu = 100
        len_layer = len(heat_dict)
        count_layer = 1
        for layer in heat_dict:
            print("The " + str(count_layer) + "th layer in " + str(len_layer) + " layers")
            self.layer_neuron_wise_mcmc(heat_dict, layer, max_neu, layer_all_info)
            count_layer = count_layer + 1

    def save_mcmc(self,layer_all_info):
        with open("metadata/"+ str(self.round) + "/mcmc_info_" + str(self.dataset) + "_" + str(self.model) + "_" + str(self.attack) + "_" + str(self.sample_size) + "_" + str(self.epsilon) + ".txt", "wb") as fp:
            pickle.dump(layer_all_info, fp, protocol=pickle.HIGHEST_PROTOCOL)

    #compute dive by
    def compute_all_divergence(self,model_target):
        print("start computing differences")
        dateTimeObj = datetime.now()
        date_name = str(dateTimeObj).replace(' ', '-').replace(':', '-').replace('.', '-')
        # print("advset[0]: " + str(advset[0]))
        # return
        heat_dict = {}
        p_value = {}
        row_max = int(self.row_max)
        row_count = 0
        act_list_len = 0
        print("len(advset): " + str(len(self.advset)))
        print("advset: " + str(self.advset))
        # for i in range(len(self.advset)):
        for i in range(self.sample_size):#for each instance in the dataset
            # print("i is: " + str(i))
            d_adv, t_adv = self.advset[i][0].to(self.device), torch.tensor(self.advset[i][1]).to(self.device)
            # pgd attack
            X_adv, y_adv = Variable(d_adv), Variable(t_adv)
            out_adv, act_list_adv = model_target.forward_act(X_adv.unsqueeze(0))
            #============only consider missclassified data=============
            corr_adv = out_adv.data.max(1)[1] == y_adv.data
            # if not corr_adv:
            #     d_test, t_test = self.testset[i][0].to(self.device), torch.tensor(self.testset[i][1]).to(self.device)
            #     # pgd attack
            #     X_test, y_test = Variable(d_test), Variable(t_test)
            #     self.compute_individual_divergence(heat_dict,model_target,X_test,y_test,row_count,act_list_adv)
            # # ======End of only consider missclassified data=============

            #===========Consider all classifications===============
            d_test, t_test = self.testset[i][0].to(self.device), torch.tensor(self.testset[i][1]).to(self.device)
                # pgd attack
            X_test, y_test = Variable(d_test), Variable(t_test)
            self.compute_individual_divergence(heat_dict,model_target,X_test,y_test,row_count,act_list_adv)
            # =======End if Consider all classifications===========
            row_count = row_count + 1
        layer_all_info = {}
        self.pre_mcmc(heat_dict,layer_all_info)
        self.save_mcmc(layer_all_info)
        # print("all info: " + str(layer_all_info))
        # converged_gr = layer_gr[layer_gr < 1.2]
        # print("layer_mean: " + str(layer_mean))
        # print("converged_gr: " + str(converged_gr))
        # print("flatten_layer_gr: " + str(layer_gr))
        # print("proportion of convegence: " + str(len(converged_gr)/len(layer_gr)))
        # assert False, "print list"
        # return
        return heat_dict, act_list_len, row_max, row_count, date_name

    def plot_heatmap(self,heat_dict, act_list_len, row_max, date_name):
        print("plotting heat map")
        for j in range(act_list_len):
            fig, ax = plt.subplots()
            # plt.imshow(np.reshape(flatten_np, (-1, len(flatten_np))), cmap='hot', interpolation='nearest')
            ax = sns.heatmap(heat_dict[j])
            plt.savefig("fig/" + self.args.dataset + "_" + self.is_black + "_" + self.attack + str(j) + "," + str(
                row_max) + "," + date_name[:-7] + ".pdf")

    def print_var(self,heat_dict, row_max, date_name):
        for j in range(1):
            # for j in range(act_list_len):
            print("plotting " + str(j))
            df = pd.DataFrame(heat_dict[j])
            if self.args.bar:
                df_var = df.var().to_frame()
                fig, ax = plt.subplots()
                ax = df_var.hist(bins=12)
                plt.savefig(
                    "fig/bin_" + self.args.dataset + "_10_" + str(j) + "," + str(row_max) + "," + date_name[
                                                                                                  :-7] + ".pdf")
            else:
                print("combing df and comparing ranks")
                all_df_rank = []
                all_df_var = []
                h_rows = np.linspace(row_max / 5, row_max, 5, dtype=int)
                for h_row in h_rows:
                    df_var = df.head(h_row).var().to_frame()
                    df_var.columns = ['a_' + str(h_row)]
                    df_rank = df_var.rank()
                    df_rank.columns = ['a_' + str(h_row)]
                    # df_var.sort_values(by=['a_' + str(h_row)], ascending=False)
                    all_df_var.append(df_var)
                    all_df_rank.append(df_rank)
                show_df_rank = pd.concat(all_df_rank, axis=1)
                show_df_var = pd.concat(all_df_var, axis=1)
                show_df_rank_sort = show_df_rank.sort_values(by=['a_' + str(h_rows[0])],
                                                             ascending=False)
                show_df_var_sort = show_df_var.sort_values(by=['a_' + str(h_rows[0])], ascending=False)
                show_df_rank_sort.to_csv("fig/" + self.args.dataset + "adv_rank_" + str(row_max) + "," + str(
                    date_name[:-7]) + ".csv", index=True, header=True)
                show_df_var_sort.to_csv("fig/" + self.args.dataset + "adv_var_" + str(row_max) + "," + str(
                    date_name[:-7]) + ".csv", index=True, header=True)
                print(show_df_rank_sort)
                print(show_df_var_sort)
                # fig, ax = plt.subplots()
                # ax = df_var.hist(bins=12)
            # df_var.columns = ['a']
            # print(df_var.sort_values(by=['a'],ascending=False))
            # plt.savefig("fig/cifar_10_" + str(j) + "," + str(row_max) + "," + date_name + ".pdf")

    def eval_acc(self,model,test_loader,adv_loader,device):
        """
        evaluate model by white-box attack
        """
        model.eval()
        robust_err_total = 0
        natural_err_total = 0
        natural_total = 0
        robust_total = 0

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # pgd attack
            err_natural = (model(data).data.max(1)[1] != target.data).float().sum()
            natural_err_total += err_natural
            natural_total += (target.data == target.data).float().sum()


        for data, target in adv_loader:
            data, target = data.to(device), target.to(device)
            # pgd attack
            err_robust = (model(data).data.max(1)[1] != target.data).float().sum()
            robust_err_total += err_robust
            robust_total += (target.data == target.data).float().sum()

        print('natural_err_total: ' + str(natural_err_total/natural_total*100) + "%")
        print('robust_err_total: ' + str(robust_err_total/robust_total*100) + "%")


    def eval_adv_test_blackbox(self,model_target, model_source, test_loader):
        """
        evaluate model by white-box attack
        """
        model_target.eval()
        # model_source.eval()
        #Generating dataset
        if self.args.gen:
            self.gen_data(model_target, test_loader)
        #computing diff
        elif self.args.diff:
            heat_dict, act_list_len, row_max, row_count, date_name = self.compute_all_divergence(model_target)
            if row_count == row_max:
                if self.args.heat:
                    self.plot_heatmap(heat_dict, act_list_len, row_max, date_name)
                if self.args.var:
                    self.print_var(heat_dict, row_max, date_name)
        else:
            assert False,"please choose a correct analytic action"
        # else:
        #     robust_err_total = 0
        #     natural_err_total = 0
        #     for data, target in test_loader:
        #         print("start the original attack")
        #         data, target = data.to(device), target.to(device)
        #         # pgd attack
        #         X, y = Variable(data, requires_grad=True), Variable(target)
        #         err_natural, err_robust = self._pgd_blackbox(model_target, model_source, X, y,self.args.epsilon,self.args.num_steps,self.args.step_size)
        #         robust_err_total += err_robust
        #         natural_err_total += err_natural
        #     print('natural_err_total: ', natural_err_total)
        #     print('robust_err_total: ', robust_err_total)