import subprocess

import numpy as np

DATASET = ['citeseer', 'cora']
NNPU = [True, False]
P = list(range(1, 6))
BIAS = [True, False]
SKIP_CONN = [True, False]
DROPOUT = list(np.arange(1, 9, 0.5))

DATASET = ['cora']
NNPU = [True]
P = [1]

for data in DATASET:
    for nnpu in NNPU:
        for p_ in P:
            for d1 in DROPOUT:
                for d2 in DROPOUT:
                    command = "python pyg.py --dataset={} --p={} ".format(data, p_/100)
                    if nnpu: command += " --nnpu "
                    command += "--d1={} ".format(d1/10)
                    command += "--d2={} ".format(d2/10)
                    # print(command)
                    subprocess.run(["echo", command])
                    subprocess.run(command.split())
