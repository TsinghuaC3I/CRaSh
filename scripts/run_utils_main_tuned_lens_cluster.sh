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
        bs=4
    elif [ $model == "60b" ]
    then
        bs=1
    fi
    
    # ## full vs full
    # --datasets_to_cluster "arc_easy,arc_challenge,web_questions,openbookqa,piqa,hellaswag,sciq,race" \
    for TASK in "wikitext" #"arc_easy" "arc_challenge" "web_questions" "openbookqa" "piqa" "hellaswag" "sciq" "race"
    do
        for similarity_type in 'wasserstein_distance' 'hellinger_distance' 'js_div' "kl_div" 'cosine_similarity'
        do
            for num_samples in 128 256 512 1024
            do
                echo "task: ${TASK}, model: ${MODEL}, similarity_type: ${similarity_type}, num_samples: ${num_samples}"
                CUDA_VISIBLE_DEVICES="0" python main.py \
                --model_name_or_path "/root/pubmodels/transformers/$MODEL" \
                --dataset_name ${TASK} \
                --similarity_type ${similarity_type} \
                --datasets_to_cluster "copa,trivia_qa,boolq,story_cloze,arc_easy,arc_challenge,web_questions,openbookqa,piqa,hellaswag,sciq,race" \
                --per_device_train_batch_size $bs \
                --per_device_eval_batch_size $bs \
                --additional_note "N${num_samples}" \
                --visual_type "tuned_lens_similarity" \
                --fig_dir "/root/kyzhang/studio/transfer_llm/figs_cluster" \
                --validation_num_samples ${num_samples}
            done
        done
    done
done
