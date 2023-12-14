#!/usr/bin/env python
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --config_file accelerate_config.yaml ./training/run_pseudo_labelling.py \
  --model_name_or_path "/data1/yumingdong/model/huggingface/whisper-large-v3/" \
  --dataset_name "/data1/yumingdong/whisper/distil-whisper/dataset/test_hk_can/" \
  --dataset_config_name "" \
  --dataset_split_name "train" \
  --text_column_name "sentence" \
  --id_column_name "id" \
  --output_dir "./Cantonese" \
  --per_device_eval_batch_size 16 \
  --dtype "float16" \
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