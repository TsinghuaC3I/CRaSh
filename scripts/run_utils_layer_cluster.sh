MODEL="facebook/opt-1.3b"
bs=32

# ## full vs full
for TASK in "wikitext" "arc_easy" "arc_challenge" "openbookqa" "web_questions" "piqa" "hellaswag" "sciq" "race"
do
    for pooling in "mean" "max" "max_nonpad" "last" "mean_all"
    do
        CUDA_VISIBLE_DEVICES=0 python utils/main.py \
        --model_name_or_path "$MODEL" \
        --dataset_name ${TASK} \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --pooling ${pooling} \
        --num_keep_layers 12 \
        --visual_type "layer_iter_cluster" \
        --num_layer_cluster 12 \
        --cka_minibatch 0 \
        --additional_note "${MODEL}-${pooling}" \
        --validation_num_samples 512

        CUDA_VISIBLE_DEVICES=0 python utils/main.py \
        --model_name_or_path "$MODEL" \
        --dataset_name ${TASK} \
        --per_device_train_batch_size $bs \
        --per_device_eval_batch_size $bs \
        --pooling ${pooling} \
        --num_keep_layers 12 \
        --visual_type "layer_iter_cluster" \
        --num_layer_cluster 12 \
        --cka_minibatch 0 \
        --additional_note "${MODEL}-${pooling}-only_context" \
        --validation_num_samples 512 \
        --only_context
    done
done
