# tasks=(down_sampling gaussian_blur motion_blur colorization inpainting_rand)
tasks=(down_sampling)
sampler=edm_fig
data=celeba

for task in "${tasks[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python FIG_diff/posterior_sample.py \
    +data=$data \
    +model=celeba \
    +task=$task \
    +sampler=$sampler \
    save_dir=./FIG_diff/results_${data} \
    num_runs=1 \
    batch_size=1 \
    data.start_id=0 data.end_id=1 \
    name=${task}_${sampler} \
    gpu=0 \
    seed=45
done
