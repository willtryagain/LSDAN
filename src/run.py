import subprocess

import numpy as np

DATASET = ['citeseer', 'cora']
NNPU = [True, False]
P = list(range(1, 6))
BIAS = [True, False]
SKIP_CONN = [True, False]
DROPOUT1 = list(np.arange(1, 9, 0.5))
DROPOUT2 = list(np.arange(1, 9, 0.5))
# DROPOUT3 = list(np.arange(1, 9, 0.5))
# DROPOUT4 = list(np.arange(1, 9, 0.5))


DATASET = ['citeseer']
NNPU = [True]
P = [1, 2]

for data in DATASET:
    for nnpu in NNPU:
        for p_ in P:
            for d1 in DROPOUT1:
                for d2 in DROPOUT2:
                    command = "python pyg.py --dataset={} --p={} ".format(data, p_/100)
                    if nnpu: command += " --nnpu "
                    command += "--d1={} ".format(d1/10)
                    command += "--d2={} ".format(d2/10)
                    # print(command)
                    subprocess.run(["echo", command])
                    subprocess.run(command.split())
