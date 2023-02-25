import argparse

import numpy as np

DATASET = ['citeseer', 'cora']
NNPU = [True, False]
P = list(range(1, 6))
DROPOUT = np.arange(0.5, 0.75, 0.05)
LAMBDA = [0.1, 0.01, 0.001, 0.0001, 0.00001]

DATASET = ['cora']
NNPU = [True]
P = list(range(2, 6))


parser = argparse.ArgumentParser(description='PyTorch LSDAN')
parser.add_argument('--loc', default='all_gath.txt', type=str)
parser.add_argument('--i', default=96, type=int)
parser.add_argument('--n', default=10000, type=int)
parser.add_argument('--type', default='avg', type=str)


args = parser.parse_args()


index = args.i

with open(args.loc, "r") as f:
    flines = f.readlines()

lines = []
for i, line in enumerate(flines):
    if i < index or len(line.split()) != 1:
        lines.append(line)

if args.type != 'avg':
    seeds = []
    mx = 0

    for line in lines[index:]:
        # split line and convert to float
        line = line.split()
        if float(line[0]) > 0.76:
            mx = float(line[0])
            seed = int(line[1])
            seeds.append(seed)
    print(seeds)


LIMS = [0.808, 0.825, 0.838, .845]
if args.type == 'avg':
    for data in DATASET:
        for nnpu in NNPU:
            for p_ in P:
                mx = 0
                var = 0
                seeds_n = ""
                seeds = []
                index += 1
                for _ in range(args.n):
                    line = lines[index].split()
                    if float(line[0]) > LIMS[p_ - 2]:
                        mx = float(line[0])
                        seeds_n = int(line[1])
                        seeds.append(seeds_n)

                    index += 1
                index += 1
                print("dataset: {}, nnpu: {}, p: {}, mx: {}".format(data, nnpu, p_, mx))
                print(seeds)