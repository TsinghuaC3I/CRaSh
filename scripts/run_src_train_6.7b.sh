export NCCL_P2P_LEVEL=NVL

MODEL="facebook/opt-1.3b"

selected_layers=("openbookqa mean_weighted 0,8,17,31 1,2,3,5,6,7,9,10,11,12,13,15,20,22,23,24,27,30" \
    "piqa mean_weighted 0,8,17,31 1,2,3,5,6,7,9,10,11,12,13,15,20,22,23,24,27,30" \
    "sciq mean_weighted 0,8,17,31 1,2,3,5,6,7,9,10,11,12,13,15,20,22,23,24,27,30" \
    "race mean_weighted 0,8,17,31 1,2,3,5,6,7,9,10,11,12,13,15,20,22,23,24,27,30" \
    "arc_easy mean_weighted 0,9,16,31 1,2,3,5,7,8,10,11,12,13,14,15,19,22,24,25,28,30" \
    "arc_challenge mean_weighted 0,9,16,31 1,2,3,5,7,8,10,11,12,13,14,15,19,22,24,25,28,30" \
    "web_questions mean_weighted 0,9,16,31 1,2,3,5,7,8,10,11,12,13,14,15,19,22,24,25,28,30" \
    "hellaswag mean_weighted 0,7,14,31 1,2,3,4,5,6,8,9,10,11,12,13,15,16,17,21,27,30")

bs=2
eval_steps=100
for item in "${selected_layers[@]}"
do
    IFS=' ' read -ra sub_tuple <<< "$item"
    
    TASK=${sub_tuple[0]}
    pooling=${sub_tuple[1]}
    train_layers=${sub_tuple[2]}
    student_layers=${sub_tuple[3]}
    
    for lr in 8e-6 2e-5 5e-5; do
        echo ">>>>>>>>>>>>> TASK = ${TASK}, pooling = ${pooling} lr = ${lr} <<<<<<<<<<<<<<<"
        echo "student_layers = ${student_layers}, train_layers = ${train_layers}"
        ### emulator ft
        CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
        --mixed_precision=bf16 --main_process_port 29800 --multi_gpu \
        src/run_train_custom.py \
        --model_name_or_path $MODEL \
        --dataset_name ${TASK} \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --learning_rate $lr \
        --lr_scheduler_type cosine \
        --num_train_epochs 3 \
        --train_layers ${train_layers} \
        --student_layers ${student_layers} \
        --num_warmup_steps 20 \
        --seed 42 \
        --eval_steps $eval_steps \
        --output_dir logs/table_cluster/task_support/${MODEL}/${TASK}/${pooling}/${train_layers}/${student_layers}/${lr}

        # bs=4
        # eval_steps=20
        CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
        --mixed_precision=bf16 --main_process_port 29800 --multi_gpu \
        src/run_train_custom.py \
        --model_name_or_path $MODEL \
        --dataset_name ${TASK} \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --learning_rate $lr \
        --lr_scheduler_type cosine \
        --num_train_epochs 3 \
        --train_layers ${train_layers} \
        --student_layers ${student_layers} \
        --seed 42 \
        --student_strategy "repeat" \
        --num_warmup_steps 20 \
        --eval_steps $eval_steps \
        --output_dir logs/table_cluster/task_support/${MODEL}/${TASK}/${pooling}-repeat-all/${train_layers}/${student_layers}/${lr}

        # bs=4
        # eval_steps=20
        CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
        --mixed_precision=bf16 --main_process_port 29800 --multi_gpu \
        src/run_train_custom.py \
        --model_name_or_path $MODEL \
        --dataset_name ${TASK} \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --learning_rate $lr \
        --lr_scheduler_type cosine \
        --num_train_epochs 3 \
        --train_layers ${train_layers} \
        --student_layers ${student_layers} \
        --seed 42 \
        --only_repeat_non_trainable_layers \
        --student_strategy "repeat" \
        --num_warmup_steps 20 \
        --eval_steps $eval_steps \
        --output_dir logs/table_cluster/task_support/${MODEL}/${TASK}/${pooling}-repeat-non-trainable/${train_layers}/${student_layers}/${lr}
    done
done
