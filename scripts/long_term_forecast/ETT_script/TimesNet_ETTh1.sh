
## TIMESNET MODEL - Long-Term Forecasting on ETTh1
## TimesNet uses FFT to discover periods and 2D convolutions
## to capture intraperiod and interperiod variations.
##
## Course connection: Convolutional Neural Networks (CNN)
## - 2D convolutions on reshaped time series
## - Residual connections


# export CUDA_VISIBLE_DEVICES=2

model_name=TimesNet

## RUN 1: PRED_LEN = 96

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --train_epochs 10 \
  --num_workers 4 \
  --use_gpu

# RUN 2: PRED_LEN = 192

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --train_epochs 10 \
  --num_workers 4 \
  --use_gpu