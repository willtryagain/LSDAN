import subprocess

for data in ['cora', 'citeseer']:
    for nnpu in [True, False]:
        for p_ in range(1, 6):
            for bias in [True, False]:  
                for add_skip_connection in [True, False]:
                        for d in range(1, 11):
                            command = "python main.py --dropout={} --dataset={} --p={}".format(d/10, data, p_/100)
                            if bias: command += " --bias "
                            if add_skip_connection: command += " --add_skip_connection "
                            if nnpu: command += " --nnpu "
                            print(command)
                            subprocess.run(["echo", command])
                            subprocess.run(command.split())
