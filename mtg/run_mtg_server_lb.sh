NUM_WORKERS=${NUM_WORKERS:-3}
MAIN_PORT=${REWARD_PORT:-8098}
echo "Main port for Load Balancer: $MAIN_PORT"
echo "Number of workers: $NUM_WORKERS"

# Workers listen on MAIN_PORT+1, MAIN_PORT+2, ..., MAIN_PORT+NUM_WORKERS
for cnt in $(seq 1 $NUM_WORKERS)
do
    CUDA_VISIBLE_DEVICES=0 python mtg_server_lb.py --port $((MAIN_PORT + cnt)) &
done

# Load balancer (foreground)
python balancer.py --port $MAIN_PORT
