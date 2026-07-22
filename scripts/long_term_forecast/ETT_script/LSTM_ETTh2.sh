
## LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN)
## designed to capture long-term dependencies in sequential data.
## explainantion of each term in bottom 

# export CUDA_VISIBLE_DEVICES=2


model_name=LSTM


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_96 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --d_model 16 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 16 \
  --train_epochs 5 \
  --num_workers 0 \
  --no_use_gpu


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_96_192 \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 7 \
  --c_out 7 \
  --d_model 16 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 16 \
  --train_epochs 5 \
  --num_workers 0 \
  --no_use_gpu

# COMMENTED OUT: OTHER PREDICTION HORIZONS

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_336 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 336 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --c_out 7 \
#   --d_model 16 \
#   --des 'Exp' \
#   --itr 1 \
#   --batch_size 16 \
#   --train_epochs 5 \
#   --num_workers 0 \
#   --no_use_gpu

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh2.csv \
#   --model_id ETTh2_96_720 \
#   --model $model_name \
#   --data ETTh2 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 720 \
#   --e_layers 2 \
#   --enc_in 7 \
#   --c_out 7 \
#   --d_model 16 \
#   --des 'Exp' \
#   --itr 1 \
#   --batch_size 16 \
#   --train_epochs 5 \
#   --num_workers 0 \
#   --no_use_gpu





##=============================
## explanation of each terms 
# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \                     ## training = 1, test =0
#   --root_path ./dataset/ETT-small/ \    ## root
#   --data_path ETTh2.csv \               
#   --model_id ETTh2_96_96 \              ## experiment Id: dataset, length, pred-lenght
#   --model $model_name \
#   --data ETTh2 \
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
#   --batch_size 16 \
#   --train_epochs 3 \
#   --num_workers 0 \
#   --no_use_gpu