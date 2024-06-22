#export CUDA_DEVICE_ORDER="PCI_BUS_ID"
#export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO
#nohup python main.py\
#    --train_data ./data/train.txt\
#    --dev_data ./data/dev.txt\
#    --tgt_vocab_f ./data/vocab.txt\
#    --ckpt_path ./ckpt/ > output_train.log 2>&1 &
python main.py\
    --train_data ./data/train.txt\
    --dev_data ./data/dev.txt\
    --tgt_vocab_f ./data/vocab.txt\
    --ckpt_path ./ckpt/
