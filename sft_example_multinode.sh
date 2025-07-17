#!/bin/bash

# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2

MODEL=<MODEL>

COMMAND="uv run ./examples/run_sft.py --config=examples/configs/sft_llama_nemotron.yaml cluster.num_nodes=$NUM_ACTOR_NODES cluster.gpus_per_node=8 checkpointing.checkpoint_dir='results/sft_llama_nemotron_$MODEL' logger.wandb_enabled=True logger.wandb.name='sft-llama_nemotron-$MODEL'" \
CONTAINER=<CONTAINER> \
MOUNTS="<MOUNTS>" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=<ACCOUNT> \
    --job-name=<JOB_NAME> \
    --partition=<PARTITION> \
    --time=0:20:0 \
    ray.sub