MODEL="facebook/opt-1.3b"
bs=4
eval_steps=10

selected_layers=("arc_easy mean_weighted 0,5,18,23 1,2,3,9,12,15,20,22" \
    "arc_challenge mean_weighted 0,5,18,23 1,2,3,9,12,15,20,22" \
    "web_questions mean_weighted 0,5,18,23 1,2,3,9,12,15,20,22" \
    "hellaswag mean_weighted 0,5,12,23 1,2,3,8,10,11,17,22")

for item in "${selected_layers[@]}"
do
    IFS=' ' read -ra sub_tuple <<< "$item"
    
    TASK=${sub_tuple[0]}
    pooling=${sub_tuple[1]}
    train_layers=${sub_tuple[2]}
    student_layers=${sub_tuple[3]}
    
    for lr in 1e-4 2e-5 5e-5 8e-5; do
        echo ">>>>>>>>>>>>> TASK = ${TASK}, pooling = ${pooling} lr = ${lr} <<<<<<<<<<<<<<<"
        echo "student_layers = ${student_layers}, train_layers = ${train_layers}"
        ### emulator ft
        bs=6
        eval_steps=10
        CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
        --mixed_precision=bf16 --main_process_port 29600 --multi_gpu \
        src/run_train_custom.py \
        --model_name_or_path $MODEL \
        --dataset_name ${TASK} \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --learning_rate $lr \
        --lr_scheduler_type cosine \
        --num_train_epochs 5 \
        --train_layers ${train_layers} \
        --student_layers ${student_layers} \
        --seed 42 \
        --eval_steps $eval_steps \
        --output_dir logs/table_cluster/task_support/${MODEL}/${TASK}/${pooling}/${train_layers}/${student_layers}/${lr}

        bs=4
        eval_steps=20
        CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
        --mixed_precision=bf16 --main_process_port 29600 --multi_gpu \
        src/run_train_custom.py \
        --model_name_or_path $MODEL \
        --dataset_name ${TASK} \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --learning_rate $lr \
        --lr_scheduler_type cosine \
        --num_train_epochs 5 \
        --train_layers ${train_layers} \
        --student_layers ${student_layers} \
        --seed 42 \
        --student_strategy "repeat" \
        --eval_steps $eval_steps \
        --output_dir logs/table_cluster/task_support/${MODEL}/${TASK}/${pooling}-repeat-all/${train_layers}/${student_layers}/${lr}

        bs=4
        eval_steps=20
        CUDA_VISIBLE_DEVICES="0,1,2,3" accelerate launch \
        --mixed_precision=bf16 --main_process_port 29600 --multi_gpu \
        src/run_train_custom.py \
        --model_name_or_path $MODEL \
        --dataset_name ${TASK} \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --learning_rate $lr \
        --lr_scheduler_type cosine \
        --num_train_epochs 5 \
        --train_layers ${train_layers} \
        --student_layers ${student_layers} \
        --seed 42 \
        --only_repeat_non_trainable_layers \
        --student_strategy "repeat" \
        --eval_steps $eval_steps \
        --output_dir logs/table_cluster/task_support/${MODEL}/${TASK}/${pooling}-repeat-non-trainable/${train_layers}/${student_layers}/${lr}
    done
done
