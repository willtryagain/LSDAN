import subprocess



DATASET = ['citeseer', 'cora']
NNPU = [True, False]
P = list(range(1, 6))
BIAS = [True, False]
SKIP_CONN = [True, False]
DROPOUT = list(range(1, 10))

DATASET = ['cora']
NNPU = [False]
P = [5]

for data in DATASET:
    for nnpu in NNPU:
        for p_ in P:
            for bias in BIAS:  
                for add_skip_connection in SKIP_CONN:
                        for d in DROPOUT:
                            command = "python main.py --dataset={} --p={} ".format(data, p_/100)
                            if nnpu: command += " --nnpu "
                            if bias: command += " --bias "
                            if add_skip_connection: command += " --add_skip_connection "
                            command += "--dropout={} ".format(d/10)
                            # print(command)
                            subprocess.run(["echo", command])
                            subprocess.run(command.split())
