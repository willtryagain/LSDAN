import subprocess

import numpy as np

DATASET = ['cora']
NNPU = [True]
P = list(range(2, 6))
for data in DATASET:
    for nnpu in NNPU:
        for p_ in P:
                command = "python pyg.py --dataset={} --p={} ".format(data, p_/100)
                if nnpu: command += " --nnpu "
                subprocess.run(["echo", command])
                for _ in range(1):
                    subprocess.run(command.split())
