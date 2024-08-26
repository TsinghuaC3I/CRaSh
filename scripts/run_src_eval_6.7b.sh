MODEL="facebook/opt-1.3b"

selected_layers=("arc_easy mean_weighted 0,8,19,31 1,2,3,5,6,7,11,13,14,15,16,18,20,22,23,24,27,30" \
    "arc_challenge mean_weighted 0,8,19,31 1,2,3,5,6,7,10,13,14,15,16,18,20,22,23,24,27,30" \
    "web_questions mean_weighted 0,7,14,31 1,2,3,4,5,6,8,9,10,11,12,13,15,16,19,23,28,30" \
    "openbookqa mean_weighted 0,15,22,31 1,2,3,8,13,14,16,17,18,19,20,21,23,24,25,27,29,30")

output_dir="logs/table_cluster/task_specific"

for item in "${selected_layers[@]}"
do
    IFS=' ' read -ra sub_tuple <<< "$item"
    
    TASK=${sub_tuple[0]}
    pooling=${sub_tuple[1]}
    train_layers=${sub_tuple[2]}
    student_layers=${sub_tuple[3]}

    for lr in 8e-6 2e-5 5e-5; do
        echo ">>>>>>>>>>>>> TASK = ${TASK}, pooling = ${pooling}, note = ${note}, lr = ${lr} <<<<<<<<<<<<<<<"
        echo "student_layers = ${student_layers}, train_layers = ${train_layers}"
        echo "plugin"
        note=${pooling}
        CUDA_VISIBLE_DEVICES=0 python src/run_eval_custom.py \
        --model_name_or_path $MODEL \
        --tasks ${TASK} \
        --load_adapter ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/adapter.pt \
        --output_dir ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/plugin.json

        echo "emulator"
        CUDA_VISIBLE_DEVICES=0 python src/run_eval_custom.py \
        --model_name_or_path $MODEL \
        --tasks ${TASK} \
        --load_student \
        --load_adapter ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/adapter.pt \
        --output_dir ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/emulator.json

        note="mean_weighted-repeat-non-trainable"
        echo "emulator, mean_weighted-repeat-non-trainable"
        CUDA_VISIBLE_DEVICES=0 python src/run_eval_custom_0610.py \
        --model_name_or_path $MODEL \
        --tasks ${TASK} \
        --student_strategy "repeat" \
        --only_repeat_non_trainable_layers \
        --load_adapter ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/adapter.pt \
        --output_dir ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/emulator-repeat-non-trainable.json

       CUDA_VISIBLE_DEVICES=0 python src/run_eval_custom.py \
        --model_name_or_path $MODEL \
        --tasks ${TASK} \
        --load_adapter ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/adapter.pt \
        --output_dir ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/plugin.json

        note="mean_weighted-repeat-all"
        echo "emulator, mean_weighted-repeat-all"
        CUDA_VISIBLE_DEVICES=0 python src/run_eval_custom_0610.py \
        --model_name_or_path $MODEL \
        --tasks ${TASK} \
        --student_strategy "repeat" \
        --load_adapter ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/adapter.pt \
        --output_dir ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/emulator-repeat-all.json

        CUDA_VISIBLE_DEVICES=0 python src/run_eval_custom.py \
        --model_name_or_path $MODEL \
        --tasks ${TASK} \
        --load_adapter ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/adapter.pt \
        --output_dir ${output_dir}/${MODEL}/${TASK}/${note}/${train_layers}/${student_layers}/${lr}/plugin.json
    done
done