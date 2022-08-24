import subprocess
import time
attack = "rand"
round = 3
dataset_map = {
    "wideresnet":"cifar10",
    "smallcnn":"mnist",
    "densenet121":"cifar10",
    "fcnet5":"mnist",
    "fcnet10":"mnist",
    "conv1dnet":"mnist",
    "conv2dnet":"mnist"
}
f = open("time.txt", "a")
for ss in [10,500,1000]:
    for model in ["densenet121","fcnet5","fcnet10","conv1dnet","conv2dnet"]:
        for e in [0.001,0.01,0.05,0.1,0.3,0.6,0.9,1.2]:
            for r in range(1,round+1):
                start = time.time()
                cmd1 = "python run.py --dataset " + dataset_map[model] + " --model " + model + " --attack " + str(attack) + " --diff --epsilon " + str(e) + " --samplesize " + str(ss) + " --round " + str(r)
                process = subprocess.Popen(cmd1, shell=True)
                output, error = process.communicate()
                end = time.time()
                f.write("dataset:" + str(dataset_map[model]) + ",model:" + \
                        str(model) + ",attack:" + str(attack) +",ss:" + str(ss) + ",e:" + str(e) + ",r:" + \
                        str(r) + ",time:" + str(end-start) + "\n")
f.close()
