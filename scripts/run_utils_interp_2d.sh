MODEL="facebook/opt-1.3b"
TASK="arc_easy"
bs=32
echo "Init,CRaSh,Full"
echo "${MODEL}-${TASK}-crash-vs-full-interp-2d"
# init vs 30.2 (crash) vs 32.4 (full)
CUDA_VISIBLE_DEVICES="1" python utils/run_interp_2d.py \
    --model_name_or_path "$MODEL" \
    --load_student "logs/table1/opt/opt-1.3b/${TASK}/ft_all/1e-5/student.pt" \
    --assist_adapter "logs/table_new/opt/opt-1.3b/arc_easy/0,10,16,23/1,2,5,7,13,18,21,22/5e-5/adapter.pt" \
    --dataset_name ${TASK} \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --seed 42 \
    --validation_num_samples 256 \
    --additional_note "${MODEL}-${TASK}-crash-vs-full-interp-2d" \
    --point_strs "Init,CRaSh,Full"

echo ""
echo ""
echo "OFT,CRaSh,Full"
echo "${MODEL}-${TASK}-oft-vs-crash-vs-full-interp-2d"
#  (offsite-tuning) vs 30.2 (crash) vs 32.4 (full)
CUDA_VISIBLE_DEVICES="1" python utils/run_interp_2d.py \
    --model_name_or_path "$MODEL" \
    --load_assist_adapter_to_init "logs/table_new/opt/opt-1.3b/arc_easy/0,1,22,23/2,5,7,10,13,16,18,21/5e-5/adapter.pt" \
    --load_student "logs/table1/opt/opt-1.3b/${TASK}/ft_all/1e-5/student.pt" \
    --assist_adapter "logs/table_cluster/task_support/opt/opt-1.3b/openbookqa/mean_weighted-repeat-all/0,8,19,23/1,4,6,10,12,15,21,22/8e-5/adapter.pt" \
    --dataset_name ${TASK} \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --load_finetuned_for_assist \
    --seed 42 \
    --validation_num_samples 256 \
    --additional_note "${MODEL}-${TASK}-oft-vs-crash-vs-full-interp-2d" \
    --point_strs "OFT,CRaSh,Full"
