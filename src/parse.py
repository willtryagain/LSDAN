

loc = "op_file.txt"
lims = []

N = 1
START_INDEX = 84
with open(loc, "r") as f:
    lines = f.readlines()

    for index in range(N):
        mx = 0

        cur_index = START_INDEX + index * 512
        # print(lines[cur_index])

        for i, line in enumerate(lines[cur_index:cur_index + 512]):
            if i%2 == 0:
                line = line.split()
                if float(line[0]) > mx:
                    mx = max(mx, float(line[0]))
        lims.append(int(mx * 100) / 100)


print(lims)
with open(loc, "r") as f:
    lines = f.readlines()

    for index in range(N):
        mx = 0

        cur_index = START_INDEX + index * 512
        # print(lines[cur_index])

        for i, line in enumerate(lines[cur_index:cur_index + 512]):
            if i%2 == 0:
                line = line.split()
                if float(line[0]) >= lims[index]:
                    mx = max(mx, float(line[0]))
                    print(lines[cur_index + i-1], line)

        print(mx)
        print("over")
        

# loc = "5_cite_nnpu.txt"
# with open(loc, "r") as f:
#     lines = f.readlines()

#     lims = [0.76]

#     for index in range(1):
#         mx = 0

#         cur_index = 78 + index * 512
#         # print(lines[cur_index])

#         for i, line in enumerate(lines[cur_index:cur_index + 512]):
#             if i%2 == 0:
#                 line = line.split()
#                 if float(line[0]) > lims[index]:
#                     mx = max(mx, float(line[0]))
#                     print(lines[i-1], line)
        
#         print(mx)
#         print("over")
        

