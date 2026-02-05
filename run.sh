# clean up any previous runs
pkill -9 python*

# Test EAGLE with Mixtral on multi-NPU
# ASCEND_RT_VISIBLE_DEVICES=4,5 python -m evaluation.inference_eagle \
#       --ea-model-path /data0/weights/EAGLE-mixtral-instruct-8x7B \
#       --base-model-path /data0/weights/Mixtral-8x7B-Instruct-v0.1 \
#       --model-id mixtral-eagle-test \
#       --bench-name spec_bench \
#       --temperature 0.0 \
#       --dtype float16 \
#       --tree-choices mc_sim_7b_63 \
#       --question-begin 0 \
#       --question-end 1 2>&1 | tee server_0.log

ASCEND_RT_VISIBLE_DEVICES=4 python -m evaluation.inference_eagle \
      --ea-model-path /data0/weights/EAGLE-LLaMA3-Instruct-8B \
      --base-model-path /data0/weights/Meta-Llama-3-8B-Instruct \
      --model-id llama3-eagle-test \
      --bench-name spec_bench \
      --temperature 0.0 \
      --dtype float16 \
      --tree-choices mc_sim_7b_63 \
      --question-begin 0 \
      --question-end 1 2>&1 | tee server_0.log
