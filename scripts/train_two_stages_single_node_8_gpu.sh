#!/bin/bash
#SBATCH --job-name=div_rl
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:8               # GPUs per node
#SBATCH --cpus-per-task=16
#SBATCH --mem=640G
#SBATCH --time=48:00:00

DATE=$(date +%m%d%H%M%S)
exec > "single_node_8gpu_${DATE}.out" 2> "single_node_8gpu_${DATE}.err"

# --- 1. SET UP GLOBAL NETWORKING ---
# Get the IP of the node Slurm assigned to us
export REWARD_NODE_IP=$(hostname -I | awk '{print $1}')
echo "Reward Node IP: $REWARD_NODE_IP"

find_free_port() {
    local start=8000
    local end=12900
    local count=4  # Number of consecutive ports needed

    for ((port=start; port<=end - count + 1; port++)); do
        local found=true
        # Check the block starting at 'port'
        for ((i=0; i<count; i++)); do
            if lsof -i :$((port + i)) > /dev/null 2>&1; then
                found=false
                break
            fi
        done

        if [ "$found" = true ]; then
            echo "$port"
            return
        fi
    done

    echo "No block of $count consecutive free ports found in range $start-$end" >&2
    return 1
}

export REWARD_PORT=$(find_free_port)
echo "Selected Reward Port: $REWARD_PORT"

find_free_port_master() {
    python3 -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}
export MASTER_PORT=$(find_free_port_master)
echo "Master Port: $MASTER_PORT"

cleanup() {
    echo "Caught signal, cleaning up child processes..."
    pkill -P $$  # Kills all child processes of this script
    exit
}
trap cleanup SIGINT SIGTERM EXIT

# --- 2. START REWARD SERVER (MTG) ---
# We run this in the background (&) using its own environment
echo "Starting MTG Reward Server on GPU 0..."
(
    cd ../mind-the-glitch || exit

    echo "Cleaning up any old server processes..."
    pkill -u $USER -f "mtg_server_lb.py"
    sleep 2

    # Initialize conda for this subshell
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate mtg
    
    # Use GPU 0 only
    export CUDA_VISIBLE_DEVICES=0
    exec bash run_mtg_server_lb.sh > server_log_lb_$DATE.txt 2>&1 &
) &

# --- 3. GLOBAL WAIT (Full Stack Health Check) ---
echo "Waiting for Load Balancer ($REWARD_PORT) and all 3 Workers to initialize..."

MAX_RETRIES=60  # 10 minutes
RETRY_COUNT=0

while true; do
    ALL_READY=true

    # 1. Check the Load Balancer (Master Port)
    if ! curl -s --connect-timeout 2 --noproxy "*" "http://$REWARD_NODE_IP:$REWARD_PORT" > /dev/null; then
        ALL_READY=false
        echo "   -> Load Balancer not responding yet..."
    fi

    # 2. Check each Worker Port (+1, +2, +3)
    # Since workers are on the Master Node, we check them via localhost
    for i in 1 2 3; do
        WORKER_PORT=$((REWARD_PORT + i))
        # We use /score or / (root) depending on your FastAPI setup
        if ! curl -s --connect-timeout 2 --noproxy "*" "http://127.0.0.1:$WORKER_PORT" > /dev/null; then
            ALL_READY=false
            echo "   -> Worker on port $WORKER_PORT is still loading..."
            break # No need to check other workers if one is still down
        fi
    done

    if [ "$ALL_READY" = true ]; then
        echo "✅ SUCCESS: Load Balancer and all 3 Workers are ONLINE."
        break
    fi


    ((RETRY_COUNT++))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "❌ ERROR: Reward Stack timeout. Some workers failed to load."
        exit 1
    fi

    echo "Status: Waiting for weights ($RETRY_COUNT/$MAX_RETRIES)..."
    sleep 10
done

# --- 4. START RL TRAINING (FlowGRPO) ---
echo "Starting FlowGRPO Training on GPU 1 - 7..."
(
    cd ../flow_grpo || exit

    # RE-EXPORT HERE to be 100% sure the Python script sees it
    export REWARD_NODE_IP=$(hostname -I | awk '{print $1}')
    export no_proxy="localhost,127.0.0.1,$REWARD_NODE_IP"
    export NO_PROXY=$no_proxy
    
    echo "RL Subshell checking Reward IP: $REWARD_NODE_IP"
    
    # Load specific CUDA module and activate RL env
    source /etc/profile
    module load cuda
    echo "✅ CUDA loaded: $(which nvcc)"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate flow_grpo
    
    export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
    
    # Launch training stage 1
    accelerate launch \
        --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
        --num_processes=7 \
        --main_process_port $MASTER_PORT \
        scripts/train_flux_kontext.py \
        --config config/grpo.py:divrl_flux_kontext_syncd_7gpu_stage_1

    # After stage 1 finishes, launch stage 2
    accelerate launch \
        --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
        --num_processes=7 \
        --main_process_port $MASTER_PORT \
        scripts/train_flux_kontext.py \
        --config config/grpo.py:divrl_flux_kontext_syncd_7gpu_stage_2
)

# --- 5. CLEANUP ---
# Kill the background reward server when training ends
kill %1
echo "Training Finished. Reward server shut down."