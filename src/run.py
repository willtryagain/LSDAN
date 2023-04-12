import subprocess

import numpy as np

DATASET = ['cora']
NNPU = [True]
SIZE = [10]

for n in SIZE:
    for data in DATASET:
        for nnpu in NNPU:
            command = "python pyg.py --dataset={} --seeds={} ".format(data, n)
            if nnpu: command += " --nnpu "
            subprocess.run(["echo", command])
            for _ in range(1):
                subprocess.run(command.split())
