#!/usr/bin/env python
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --config_file accelerate_config.yaml ./training/run_distillation.py \
  --train_dataset_name "/data1/yumingdong/whisper/distil-whisper/dataset/test_hk_can/" \
  --eval_dataset_name "/data1/yumingdong/whisper/distil-whisper/dataset/test_hk_can/" \
  --overwrite_cache True \
  --preprocessing_num_workers 16 \
  --audio_column_name "audio" \
  --text_column_name "sentence" \
  --eval_text_column_name "sentence " \
  --preprocessing_only False \
  --train_split_name "train" \
  --eval_split_name "train" \
  --streaming False \
  --wer_threshold 0.3 \
  --language yue \
  --task "transcribe" \
  --wandb_project "distil" \

