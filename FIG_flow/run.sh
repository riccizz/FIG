# CelabA
# $1 recon algo ["uncond", "pseudo", "dps", "pgdm", "dmps", "fig", "fig+"]
# $2 sample algo ["ode", "sde"]
# $3 task 
# $4 sampling steps 50 for ode

# task=super_resolution_svd_ap

name=$1

if [[ ${name:0:3} == "fig" ]]; then
    name="fig"
fi

python FIG_flow/recon.py --config=./FIG_flow/configs/rectified_flow/celeba_hq_pytorch_rf_gaussian.py \
                --algo_config=./FIG_flow/configs/algo/${name}.py \
                --recon_algo=$1 \
                --sample_algo=$2 \
                --data_root=./dataset/celeba_ \
                --recon_root=./FIG_flow \
                --dataset_name=celeba \
                --task=$3 \
                --config.sampling.sample_N=$4 \
                --config.sampling.sigma_variance=0.6 \
