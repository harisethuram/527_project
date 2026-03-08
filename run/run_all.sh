# run for three model types
models=(cnn clip transformer)
batch_sizes=(128 64 32)
lrs=(1e-3 1e-4 1e-5)
for model in "${models[@]}"; do
    echo "Training model: $model"
    for lr in "${lrs[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            echo "Training with learning rate: $lr"
            sbatch run/train.sh $model $batch_size $lr
            # python train.py \
            #     --model $model \
            #     --batch_size $batch_size \
            #     --lr $lr \
            #     --epochs 1 \
            #     --output_dir /gscratch/ark/hari/527_project/models/hparam/${model}/${lr}/${batch_size}/
        done
    done
done