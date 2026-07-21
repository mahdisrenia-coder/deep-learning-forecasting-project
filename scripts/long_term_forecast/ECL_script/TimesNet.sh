
# export CUDA_VISIBLE_DEVICES=4

model_name=TimesNet

# ========== ONLY RUNNING PRED_LEN 96 ==========
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 64 \
  --d_ff 128 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 8 \
  --train_epochs 3 \
  --num_workers 0 \
  --no_use_gpu

# ========== COMMENTED OUT OTHER HORIZONS ==========
# python -u run.py ... pred_len 192 ...
# python -u run.py ... pred_len 336 ...
# python -u run.py ... pred_len 720 ...
