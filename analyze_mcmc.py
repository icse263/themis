import arviz as az
import pickle
import numpy as np
import glob
import pandas as pd

import argparse

class mcmc(object):

    def __init__(self,path):
        self.path = path
        self.mcmc_info = {}

    def load_mcmc(self):
        with open(self.path, "rb") as fp:
            self.mcmc_info = pickle.load(fp)

    def load_mcmc_robust(self):
        with open(self.path, "rb") as fp:
            self.mcmc_info = pickle.load(fp)

    def converge_ratio(self,threshold):
        all_mu = []
        all_converge_mu = []
        each_layer = []
        for layer in self.mcmc_info:
            # print("self.mcmc_info: " + str(self.mcmc_info))
            print("In the " + layer + "th layer:\n")
            table_data = self.mcmc_info[layer]["mcse_mean"].filter(like = 'mu', axis=0)
            # r_hat = table["r_hat"].filter(like = 'mu', axis=0)
            all_mu.append(table_data)
            # print("r_hat: " + str(self.mcmc_info[layer].sort_values(by=['mcse_mean'],ascending=False)))

            converged = table_data[table_data <= threshold]
            # print("converged index: " + str(converged))
            all_converge_mu.append(converged)
            proportion = len(converged)/len(table_data)
            # layer_count = layer_count + 1
            each_layer.append(proportion)
            print("The converge rate of the is " + str(proportion))

        all_mu_np = [val for sublist in all_mu for val in sublist]
        all_converge_mu_np = [val for sublist in all_converge_mu for val in sublist]

        print("len of all_mu_np: " + str(len(all_mu_np)))
        print("len of all_converge_mu_np: " + str(len(all_converge_mu_np)))
        print("The overall converge rate of the is " + str(len(all_converge_mu_np)/len(all_mu_np)))
        print("The average converge rate of the is " + str(sum(each_layer) / len(each_layer)))

class robust(object):

    def __init__(self,model,dataset,epsilon,samplesize,attack):
        self.mcmc_info = {}
        self.model = model
        self.dataset = dataset
        self.epsilon = epsilon
        self.samplesize = samplesize
        self.attack = attack

    def load_rounds(self):
        folders = glob.glob("metadata/*")
        rounds = []
        for f in folders:
            f_candidate = f.replace('metadata/','')
            if f_candidate.isnumeric():
                rounds.append(f_candidate)
        return rounds

    def load_layers(self):
        folders = glob.glob("metadata/1/chosenindex_" + \
                    self.dataset + "_" + self.model + "*")
        layers = list(set([f.split("_")[3] for f in folders]))
        return layers

    def read_file(self):
        #read chosen index, mean and var
        rounds = self.load_rounds()
        layers = self.load_layers()
        # rounds = ['1','2','3','4']#debug
        # layers = ['1']#debug
        all_mcmc_df = {}
        all_mean_df = {}
        all_var_df = {}
        all_index_df = {}
        for f in rounds:
        # for f in ['1','2','3','4','5']:#debug
        #     individual_mcmc_df = {}
            individual_mean_df = {}
            individual_var_df = {}
            individual_index_df = {}
            with open("metadata/" + f + "/mcmc_info_" + \
                      self.dataset + "_" + self.model + "_" + \
                      self.attack + "_" + \
                      self.samplesize + "_" + \
                      self.epsilon + ".txt", "rb") as fp:
                all_mcmc_df[f] = pickle.load(fp)
            print("type: " + str(type(all_mcmc_df[f])))
            for l in layers:
            # for l in ['1','2','3','4','5']:#debug
                with open("metadata/" + f +"/var_" + \
                        self.dataset + "_" + self.model + "_" +\
                        l + "_" + self.attack + "_" + \
                        self.samplesize + "_" + \
                        self.epsilon + ".txt", "rb") as fp:
                    individual_var_df[l] = pickle.load(fp)
                with open("metadata/" + f +"/mean_" + \
                        self.dataset + "_" + self.model + "_" +\
                        l + "_" + self.attack + "_" + \
                        self.samplesize + "_" + \
                        self.epsilon + ".txt", "rb") as fp:
                    individual_mean_df[l] = pickle.load(fp)
                with open("metadata/" + f +"/chosenindex_" + \
                        self.dataset + "_" + self.model + "_" +\
                        l + "_" + self.attack + "_" + \
                        self.samplesize + "_" + \
                        self.epsilon + ".txt", "rb") as fp:
                    en_individual_index_df = enumerate(pickle.load(fp))
                    individual_index_df[l] = pd.DataFrame([[neuron, rank] for rank, neuron in en_individual_index_df],columns=['neuron','rank_' + f])
            all_mean_df[f] = individual_mean_df
            all_var_df[f] = individual_var_df
            all_index_df[f] = individual_index_df
        return rounds,layers,all_mcmc_df,all_mean_df,all_var_df,all_index_df #target_folders are the number of rounds under metadata/

    def analyze_index(self,target_folders,all_index_df,layer):
        all_index_df_list = []
        # for l in layers[0]:
        for l in layer:#debug
            join_index = all_index_df[target_folders[0]][l].merge(all_index_df[target_folders[1]][l], how='left')
            for f in target_folders[2:]:
                join_index = join_index.merge(all_index_df[f][l], how='left')
                # all_index_df_list.append(all_index_df[f][l])
        # print(all_index_df)#debug
        # print(all_index_df_list)#debug
        # merged_all_index_df = pd.concat(all_index_df_list, axis=1)
        # print(merged_all_index_df)
        # print(join_index)
        return join_index

    def analyze_mean(self, target_folders,all_mean_df,layer):
        all_mean_df_list = []
        for l in layer:#debug
            for f in target_folders:
                all_mean_df_list.append(all_mean_df[f][l].sort_index())
        merged_all_mean_df = pd.concat(all_mean_df_list, axis=1)
        merged_all_mean_df['var'] = merged_all_mean_df.var(axis=1)
        print(merged_all_mean_df)

    def analyze_var(self,target_folders,all_var_df,layer):
        all_var_df_list = []
        for l in layer:#debug
            for f in target_folders:
                all_var_df_list.append(all_var_df[f][l].sort_index())
        merged_all_var_df = pd.concat(all_var_df_list, axis=1)
        merged_all_var_df['var'] = merged_all_var_df.var(axis=1)
        print(merged_all_var_df)

    def mcmc_robustness(self,layers,rounds,all_mcmc_df,all_index_df,all_mean_df ):
        all_data = {}
        for l in layers:#debug
            # print(all_mcmc_df['1']['1'])
            individual_data = {}
            # join_index = all_index_df[rounds[0]][l].merge(all_index_df[rounds[1]][l], how='left')
            # join_mcmc = all_mcmc_df[rounds[0]][l].merge(all_mcmc_df[rounds[1]][l], how='left')
            for f in rounds:
            # for f in ['1','2']:
                merged_pd = all_index_df[f][l].reset_index().merge(all_mcmc_df[f][l].reset_index(), left_index=True, right_index=True)
                individual_data[f] = merged_pd[["neuron","mean","mcse_mean"]]
                # print(merged_pd)
            all_data[l] = individual_data
        final_merge = all_data[layers[0]][rounds[0]].merge(all_data[layers[0]][rounds[1]],on="neuron")
        for f in rounds[3:]:
            final_merge = final_merge.merge(all_data[layers[0]][f],on="neuron")
        print(final_merge)
        # for key in all_data[layers[0]][2:]:

    def compute_robustness(self):
        #read all df_mean
        rounds,layers,all_mcmc_df,all_mean_df,all_var_df,all_index_df = self.read_file()
        # print(type(all_index_df))
        # self.mcmc_robustness(layers, rounds, all_mcmc_df, all_index_df, all_mean_df)
        # self.analyze_index(rounds,all_index_df,layers)
        self.analyze_mean(rounds, all_mean_df, layers)
        self.analyze_var(rounds, all_var_df, layers)
        #merge converged mu and neuron, to see whether converged mu of the same neuron is the same


if __name__ == "__main__":
    #metadata/mcmc_info1.txt
    parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
    parser.add_argument('--mode', help='mcmc or robust')
    parser.add_argument('--dataset', help='which dataset')
    parser.add_argument('--model', help='which model')
    parser.add_argument('--epsilon', help='epsilon')
    parser.add_argument('--attack', help='attack')
    parser.add_argument('--samplesize', help='epsilon')
    parser.add_argument('--path', help='input path')
    parser.add_argument('--t', help='threshold')
    # parser.add_argument('--round', help='round')
    args = parser.parse_args()
    if args.mode == "mcmc":
        print("doing mcmc")
        this_mcmc = mcmc(args.path)
        this_mcmc.load_mcmc()
        this_mcmc.converge_ratio(float(args.t))
    elif args.mode == "robust":
        print("doing robust")
        this_robust = robust(args.model,args.dataset,args.epsilon,args.samplesize,args.attack)
        this_robust.compute_robustness()
    else:
        assert False, "please choose a correct action"