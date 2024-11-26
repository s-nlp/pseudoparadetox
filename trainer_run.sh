export CUDA_VISIBLE_DEVICES=0
export NVIDIA_VISIBLE_DEVICES=0

export LANG="en_US.UTF-8"
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"

python src/bart_baseline.py --config_path configs/bart_dolphin.json --input_columns en_toxic_comment --output_columns generated_neutral_sentence

#python src/bart_baseline.py --config_path configs/bart_dolphin.json --input_columns en_toxic_comment --output_columns generated_neutral_sentence
