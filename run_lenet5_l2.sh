CUDA_VISIBLE_DEVICES=0 python train.py \
    --model LeNet-5 \
    --data_path "/project/jonmay_1426/huijuanw/data/CINIC-10" \
    --save_path "/project/jonmay_1426/huijuanw/ckpt/lenet-5-l2/" \
    --weight_decay 5e-4