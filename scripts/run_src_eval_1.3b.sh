MODEL="facebook/opt-1.3b"

selected_layers=("arc_easy mean_weighted 0,5,18,23 1,2,3,9,12,15,20,22" \
    "hellaswag mean_weighted 0,5,12,23 1,2,3,8,10,11,17,22")

output_dir="logs/table_cluster/task_support"

for item in "${selected_layers[@]}"
do
    IFS=' ' read -ra sub_tuple <<< "$item"
    
    TASK=${sub_tuple[0]}
    pooling=${sub_tuple[1]}
    train_layers=${sub_tuple[2]}
    student_layers=${sub_tuple[3]}

    for lr in 1e-4 2e-5 5e-5 8e-5; do
        echo ">>>>>>>>>>>>> TASK = ${TASK}, pooling = ${pooling}, note = ${note}, lr = ${lr} <<<<<<<<<<<<<<<"
        echo "student_layers = ${student_layers}, train_layers = ${train_layers}"
        echo "plugin"
        note=""
        CUDA_VISIBLE_DEVICES=0 python src/run_eval_custom.py \
        --model_name_or_path $MODEL \
        --tasks ${TASK} \
        --load_adapter ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/adapter.pt \
        --output_dir ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/plugin.json

        note="mean_weighted-repeat-non-trainable"
        echo "emulator, mean_weighted-repeat-non-trainable"
        CUDA_VISIBLE_DEVICES=0 python src/run_eval_custom.py \
        --model_name_or_path $MODEL \
        --tasks ${TASK} \
        --load_adapter ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/adapter.pt \
        --output_dir ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/plugin.json

        note="mean_weighted-repeat-all"
        CUDA_VISIBLE_DEVICES=0 python src/run_eval_custom.py \
        --model_name_or_path $MODEL \
        --tasks ${TASK} \
        --load_adapter ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/adapter.pt \
        --output_dir ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/plugin.json
    done
done