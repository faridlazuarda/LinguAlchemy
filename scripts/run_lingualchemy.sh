EPOCHES=10
OUT_DIR="/mnt/beegfs/farid/lingualchemy/ablation"  # set customised out_path
EVAL_DIR="/home/alham.fikri/farid/lingualchemy/outputs"
SCALE=10
# 
for MODEL_NAME in bert-base-multilingual-cased xlm-roberta-base;
do
    for VECTOR in geo syntax_average_geo syntax_average syntax_knn_geo syntax_knn_syntax_average_geo syntax_knn_syntax_average syntax_knn;
    do
        CUDA_VISIBLE_DEVICES=4,5 python -m src.lingualchemy \
        --model_name ${MODEL_NAME} --epochs ${EPOCHES}  \
        --out_path ${OUT_DIR}/massive/${MODEL_NAME}/scale${SCALE}_${VECTOR} \
        --vector ${VECTOR} --scale ${SCALE} --eval_path ${EVAL_DIR}/massive/${MODEL_NAME}_scale${SCALE} \
        --override_results
    done
done
