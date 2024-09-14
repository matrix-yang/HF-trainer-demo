export NCCL_IB_DISABLE="1"
export NCCL_P2P_DISABLE="1"

torchrun --nnodes=1 --nproc_per_node=8 main.py