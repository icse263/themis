import subprocess
import argparse
import yaml
import math
import pandas as pd

class gen_o(object):

    def __init__(self,para,all_models):
        self.para = para
        self.all_models = all_models

    def gen_adv(self):
        for model in self.all_models:
            for epsilon in [0.001,0.01,0.05,0.1,0.3,0.6,0.9,1.2]:
                cmd1 = "python run.py --dataset " + self.para[model] + \
                       " --model " + str(model) + " --attack pgd --train --gen --epsilon " + str(epsilon)
                process = subprocess.Popen(cmd1, shell=True)
                output, error = process.communicate()

    def gen_csv(self,path):
        infile = open(path,'r')
        all_rows = []
        row = []
        for line in infile:
            # print("line: " + str(line))
            if "dataset" in line:
                configs = line.split(",")
                # print("line: " + str(line))
                # print("configs: " + str(configs))
                for items in configs:
                    # print("items: " + str(items))
                    row.append(items.split(":")[1])
            elif "natural_err_total" in line:
                row.append(line.split("(")[1].split(",")[0])
            elif "robust_err_total" in line:
                # print("data: " + str(items.strip("(")[1].strip(",")[0]))
                row.append(line.split("(")[1].split(",")[0])
                all_rows.append(row)
                row = []
        df = pd.DataFrame(all_rows)
        df.to_csv(path[:-3] + "csv",index=False)


    def eval_retrain(self):
        attack = "rand"
        f = open("acc_retrain.txt", "a")
        for model in self.all_models:
            for epsilon in [0.001,0.01,0.05,0.1,0.3,0.6,0.9,1.2]:
                for num_data in [10,50,100,500,1000,5000,10000]:
                    model_path = "checkpoints/model_" + model +"_" + self.para[model] + "_" + attack + "_" + str(epsilon) + "_" + str(num_data) + "-epoch{}.pt".format(math.ceil(num_data / 1000))
                    cmd1 = "python run.py --dataset " + self.para[model] + " --model " + model + " --attack " + str(
                        attack) + " --eval --epsilon " + str(epsilon) + " --model_path " + model_path
                    process = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE, encoding='utf-8')
                    output, error = process.communicate()
                    f.write("dataset:" + str(para[model]) + ",model:" + \
                            str(model) + ",attack:" + str(attack) + ",epsilon:" + str(epsilon) + ",num_data:" + str(num_data) + "\n")
                    f.write(str(output))
        f.close()

    def retrain(self):
        for model in self.all_models:
            for epsilon in [0.001,0.01,0.05,0.1,0.3,0.6,0.9,1.2]:
                for num_data in [10,50,100,500,1000,5000,10000]:
                    cmd1 = "python run.py --dataset " + self.para[model] + " --model " + model + " --attack " \
                           + "rand" + " --epsilon " + str(epsilon) + " --num_data " + str(num_data)
                    process = subprocess.Popen(cmd1, shell=True)
                    output, error = process.communicate()

    def eval_original(self):
        f = open("acc.txt", "a")
        for model in self.all_models:
            for e in [0.001, 0.01, 0.05, 0.1, 0.3, 0.6, 0.9, 1.2]:
                for attack in ["pgd", "rand"]:
                    cmd1 = "python run.py --dataset " + self.para[model] + " --model " + model + " --attack " + str(
                        attack) + " --eval --epsilon " + str(e)
                    process = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE, encoding='utf-8')
                    output, error = process.communicate()
                    f.write("dataset:" + str(para[model]) + ",model:" + \
                            str(model) + ",attack:" + str(attack) + ",epsilon:" + str(e) + "\n")
                    f.write(str(output))
            #         break
            #     break
            # break
        f.close()

if __name__ == "__main__":
    #metadata/mcmc_info1.txt
    parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
    parser.add_argument('--mode', help='mcmc or robust')
    parser.add_argument('--path', help='file path')
    # parser.add_argument('--round', help='round')
    args = parser.parse_args()
    # all_models = ["wideresnet", "smallcnn", "densenet121", "fcnet5", "fcnet10", "conv1dnet", "conv2dnet"]
    all_models = ["densenet121", "fcnet5", "fcnet10", "conv1dnet", "conv2dnet"]
    with open(r'yaml/model_data.yaml') as file:
        para = yaml.load(file)
    gen_instance = gen_o(para,all_models)
    if args.mode == "gen":
        gen_instance.gen_adv()
    elif args.mode == "retrain":
        gen_instance.retrain()
    elif args.mode == "eval_original":
        gen_instance.eval_original()
    elif args.mode == "eval_retrain":
        gen_instance.eval_retrain()
    elif args.mode == "csv":
        gen_instance.gen_csv(args.path)
    else:
        assert False, "please choose a correct action"