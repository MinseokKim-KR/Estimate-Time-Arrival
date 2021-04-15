#!/bin/bash
args=("$@")
echo "model_NUM : " ${args[8]}
python2 main.py --task test  --batch_size ${args[0]}  --pooling_method ${args[1]} --kernel_size ${args[2]} --alpha ${args[3]} --log_file run_log --config_file ${args[4]} --result_file ${args[5]} --weight_file ${args[6]} --epochs ${args[7]} --model_num ${args[8]} --tensorboard ${args[9]} --save_result_file ${args[10]}


