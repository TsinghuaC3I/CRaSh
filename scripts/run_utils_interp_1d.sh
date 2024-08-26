MODEL="facebook/opt-1.3b"
TASK="openbookqa"
bs=32
CUDA_VISIBLE_DEVICES="2" python utils/run_interp_1d.py \
    --model_name_or_path "$MODEL" \
    --load_student "logs/table1/opt/opt-1.3b/${TASK}/ft_all/1e-5/student.pt" \
    --assist_adapter "logs/table_cluster/task_support/opt/opt-1.3b/openbookqa/mean_weighted-repeat-all/0,8,19,23/1,4,6,10,12,15,21,22/8e-5/adapter.pt" \
    --dataset_name ${TASK} \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --seed 42 \
    --validation_num_samples 256 \
    --additional_note "${MODEL}-${TASK}-crash-vs-full-interp-1d" \
    --title "opt-1.3b, ${TASK} dataset"

CUDA_VISIBLE_DEVICES="2" python utils/run_interp_1d.py \
    --model_name_or_path "$MODEL" \
    --load_student "logs/table1/opt/opt-1.3b/${TASK}/ft_all/1e-5/student.pt" \
    --assist_adapter "logs/table_new/opt/opt-1.3b/arc_challenge/0,1,22,23/2,5,7,10,13,16,18,21/5e-5/adapter.pt" \
    --dataset_name ${TASK} \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --seed 42 \
    --validation_num_samples 256 \
    --additional_note "${MODEL}-${TASK}-oft-vs-full-interp-1d" \
    --title "opt-1.3b, ${TASK} dataset"
