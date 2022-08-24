# This is the code for ICSE'23 submission
Belows are the commands to obtain the experiment results in the paper.


# convergence analysis
'''
python run.py --dataset mnist --attack rand --diff --adv --samplesize 100
--attack: type of attack [rand/pgd] (required)
--dataset: what dataset [mnist/cifar10] (required)


--model: which model [wideresnet/smallcnn/densenet121] (required)
--eval: evaluate natural accuracy and robustness accuracy/ attack (required)
--train: use training data (by default: False)
if eval:
    --epsilon: noise level (default value set to 0.031)
if not eval but retrain: --retrain
    --epsilon: noise level (default value set to 0.031)
    --num_data: number of faults to retrain (default: 1)
if not eval:
    --diff/--gen: compute divergence and mcmc/ generate data (required)
    if diff:
        --epsilon: noise level (default value set to 0.031)
        --samplesize: number of sample for mcmc (default value set to 100)
        --round: which round of testing now it is (default: 1)
        --heat/--var: output heatmap/variance of divergence (not require)
            if var:
            --row_max: (var) printing the number of rows
    if gen:#for rand attack, no need gen as rand directly transformed input dataset
        --epsilon: noise level (default value set to 0.031)

python run.py --attack rand --dataset cifar10 --model wideresnet --gen --epsilon 0.001

python run.py --attack rand --dataset cifar10 --model wideresnet --eval --epsilon 0.001

python run.py --attack pgd --dataset cifar10 --model wideresnet --retrain --epsilon 0.01 --num_data 100

python analyze_mcmc.py --mode mcmc --path metadata/1/mcmc_info_cifar10_densenet121_pgd_1000_0.05.txt --t 0.0
'''

# massively gen dataset, retrain and evaluate
gen_acc.py 
--mode: which mode [gen/retrain/eval_original/eval_original] (required)
--path: which file path (for gen_csv)
python analyze_mcmc.py:
--mode: mcmc or robust
if mcmc:
    --path: path that contains multiple test results
    --t: threshold value to identify convergence
elif robust:
    --dataset: which dataset [cifar10/mnist]
    --model: which model [wideresnet/smallcnn]
    --attack: which attack [rand/pgd]
    --epsilon: 0.01
    --samplesize: 100


............
