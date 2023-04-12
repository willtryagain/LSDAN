
for i in `seq 1 10`; do python run_benchmark.py; done
echo "Done 1"
for i in `seq 1 10`; do python run_benchmark.py --nnpu; done
echo "Done 2"
for i in `seq 1 10`; do python run_benchmark.py --dataset CiteSeer; done
echo "Done 3"
for i in `seq 1 10`; do python run_benchmark.py --dataset CiteSeer --nnpu; done
echo "Done 4"