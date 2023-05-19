
for i in `seq 1 10`; do python run_benchmark.py --method GCN+URE; done
echo "Done 1"
for i in `seq 1 10`; do python run_benchmark.py --method GCN+NRE; done
echo "Done 2"
for i in `seq 1 10`; do python run_benchmark.py --dataset CiteSeer --method GCN+URE; done
echo "Done 3"
for i in `seq 1 10`; do python run_benchmark.py --dataset CiteSeer  --method GCN+NRE;  done
echo "Done 4"