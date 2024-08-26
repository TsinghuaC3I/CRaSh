for model in "125m" "1.3b" "6.7b" "13b" "30b"
do
    MODEL="opt/opt-${model}"
    if [ $model == "125m" ]
    then
        bs=64
    elif [ $model == "350m" ]
    then
        bs=64
    elif [ $model == "1.3b" ]
    then
        bs=64
    elif [ $model == "2.7b" ]
    then
        bs=64
    elif [ $model == "6.7b" ]
    then
        bs=16
    elif [ $model == "13b" ]
    then
        bs=4
    elif [ $model == "30b" ]
    then
        bs=2
    elif [ $model == "60b" ]
    then
        bs=1
    fi
    
    # ## full vs full
    for TASK in "wikitext" "arc_easy" "arc_challenge" "web_questions" "openbookqa" "piqa" "hellaswag" "sciq" "race"
    do
        for pooling in "last" "mean" "max"
        do
            echo "task: ${TASK}, model: ${MODEL}, pooling: ${pooling}, only_context: True"
            CUDA_VISIBLE_DEVICES=1 python main.py \
            --model_name_or_path "/root/pubmodels/transformers/$MODEL" \
            --dataset_name ${TASK} \
            --per_device_train_batch_size $bs \
            --per_device_eval_batch_size $bs \
            --only_context \
            --additional_note "opt-${model}-full-vs-full-pooling${pooling}-${TASK}-bsz0-only_ctx" \
            --xlabel "layer" \
            --ylabel "layer" \
            --title "${model}, ${TASK}" \
            --pooling ${pooling} \
            --visual_type "layer_cka" \
            --fig_dir "/root/kyzhang/studio/transfer_llm/figs_l2l" \
            --cka_minibatch 0 \
            --swap_to_cpu \
            --validation_num_samples 512
        done
    done
done
