for lang in es hi ru uk zh en am ar de es hi ru uk zh 
do
    python detoxer.py --language $lang --model_path "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated" --nshot 10 --batch_size 8 \
    --output_path ./Meta-Llama-3.1-8B-Instruct_abliterated_10shot_t08_p09_from_multilingual_toxicity_dataset_toxpart_$lang.csv
done