#!/usr/bin/env python
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --config_file accelerate_config.yaml ./training/run_distillation.py \
  --model_name_or_path /data1/yumingdong/model/distiled/whisper-large-v3-en_32_de_2/ \
  --teacher_model_name_or_path /data1/yumingdong/model/huggingface/whisper-large-v3/ \
  --train_dataset_name "Cantonese" \
  --eval_dataset_name "Cantonese" \
  --overwrite_cache True \
  --preprocessing_num_workers 16 \
  --per_device_train_batch_size 8 \
  --num_train_epochs 3 \
  --logging_dir exp/logging \
  --save_total_limit 10 \
  --logging_strategy "epoch" \
  --audio_column_name "audio" \
  --text_column_name "sentence" \
  --eval_text_column_name "sentence " \
  --preprocessing_only False \
  --train_dataset_config_name '' \
  --train_split_name "train" \
  --eval_split_name "train" \
  --streaming False \
  --wer_threshold 100 \
  --language yue \
  --freeze_encoder True \
  --dtype "bfloat16" \
  --output_dir exp \
  --overwrite_output_dir \
  --do_train yes \
  --wandb_project "distil" \
  --no_streaming
