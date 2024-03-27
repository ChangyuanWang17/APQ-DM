CUDA_VISIBLE_DEVICES=4 python -u main.py \
    --config celeba.yml \
    --exp {workspace} \
    --doc {pretrain model} \
    --sample --fid --timesteps 100 --eta 0 --ni \
    --image_folder {folder name} \
    --skip_type quad \
    --bitwidth 6 \
    --calib_t_mode diff \