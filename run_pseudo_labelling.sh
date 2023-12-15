#!/usr/bin/env python
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --config_file accelerate_config.yaml ./training/run_pseudo_labelling.py \
  --model_name_or_path "/data1/yumingdong/model/finetuned/whisper-large-v3-lora100+100-final/" \
  --dataset_name "/data1/yumingdong/whisper/distil-whisper/dataset/cantonese_50h/" \
  --dataset_config_name "" \
  --dataset_split_name "train" \
  --text_column_name "sentence" \
  --id_column_name "id" \
  --output_dir "/data1/yumingdong/whisper/distil-whisper/pseudo_dataset/cantonese_50h" \
  --per_device_eval_batch_size 16 \
  --dtype "bfloat16" \
  --dataloader_num_workers 16 \
  --preprocessing_num_workers 16 \
  --logging_steps 500 \
  --max_label_length 128 \
  --language "yue" \
  --task "transcribe" \
  --return_timestamps \
  --streaming False \
  --generation_num_beams 1 \
  --decode_token_ids False \
  --attn_type 'flash_attn_2' \
#  --report_to "wandb" \
#  --wandb_project "" \
#  --push_to_hub