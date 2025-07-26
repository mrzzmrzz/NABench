
export reference_sheet="/home/ma_run_ze/lzm/rnagym/fitness/reference_sheet_final.csv"
export output_scores_dir="/home/ma_run_ze/lzm/rnagym/fitness/scores/alphagenome/"
export dms_data_dir="/home/ma_run_ze/lzm/rnagym/fitness/fitness_processed_assays"

mkdir -p "$output_scores_dir"
touch "$output_scores_dir/correlation_summary.txt"
# Get the current index from the array (0-31)
export CUDA_VISIBLE_DEVICES=7
export ALPHAGENOME_API_KEY="AIzaSyBPEJLKiiqpBDl8_nT56t46yoF1m8RoGo0"

python run.py \
    --reference_sheet  $reference_sheet \
    --task_id 45 \
    --dms_directory $dms_data_dir \
    --output_directory $output_scores_dir