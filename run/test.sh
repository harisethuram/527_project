# run for three model types
models=(clip)
batch_size=128
epochs=1
for model in "${models[@]}"; do
    echo "Training model: $model"
    python train.py \
        --model $model \
        --batch_size $batch_size \
        --epochs $epochs \
        --output_dir /gscratch/ark/hari/527_project/models/init/${model}/
done