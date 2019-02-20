export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
basepath=$(cd `dirname $0`; pwd)
echo $basepath
nohup python -u main.py --config ./Config/config.cfg > log 2>&1 &
tail -f log 
 


