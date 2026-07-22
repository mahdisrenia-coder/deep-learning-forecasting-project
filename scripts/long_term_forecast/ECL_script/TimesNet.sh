## ============================================================
## TIMESNET MODEL - Electricity (ECL) Dataset
## ============================================================

# export CUDA_VISIBLE_DEVICES=4   # Commented out for Colab

model_name=TimesNet

# ============================================================
# PRED_LEN = 96
# ============================================================
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
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --use_gpu

# ============================================================
# PRED_LEN = 192
# ============================================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --use_gpu

# ============================================================
# PRED_LEN = 336
# ============================================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --use_gpu

# ============================================================
# PRED_LEN = 720
# ============================================================
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1 \
  --use_gpu



##=============================
## explanation of each terms 
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \                     ## training = 1, test =0
#   --root_path ./dataset/ETT-small/ \    ## root
#   --data_path ETTh1.csv \               
#   --model_id ETTh1_96_96 \              ## experiment Id: dataset, length, pred-lenght
#   --model $model_name \
#   --data ETTh1 \
#   --features M \                        ## multivariate(predicts all the features)
#   --seq_len 96 \
#   --label_len 48 \                      ## Overlap between input and output
#   --pred_len 96 \
#   --e_layers 2 \                        ## Number of LSTM layers (stacked)
#   --d_layers 1 \                        ## Number of decoder features(not usefull)
#   --factor 3 \                          ## Expansion factor (not critical for LSTM)
#   --enc_in 7 \                          ## Number of input features
#   --dec_in 7 \
#   --c_out 7 \                           ## Number of output features
#   --d_model 16 \                        ##LSTM hidden size
#   --d_ff 32 \                           ## Feed-forward dimension (compatibility)
#   --des 'Exp' \                         ##description
#   --itr 1 \
#   --top_k 5 \