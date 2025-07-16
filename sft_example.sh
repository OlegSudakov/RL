#!/bin/bash

uv run python examples/run_sft.py --config=examples/configs/sft_llama_nemotron.yaml cluster.gpus_per_node=8 policy.model_name=meta-llama/Llama-3.1-8B-Instruct