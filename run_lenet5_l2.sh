CUDA_VISIBLE_DEVICES=0 python train.py \
    --model LeNet-5 \
    --data_path "./data/CINIC-10" \
    --save_path "./ckpt/lenet-5-l2/" \
    --weight_decay 5e-4
