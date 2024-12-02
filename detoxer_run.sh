for lang in en
do
    python detoxer.py --language $lang --model_path "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated" \
    --nshot 10 --batch_size 8 \
    --output_path ./Meta-Llama-3.1-8B-Instruct_abliterated_10shot_$lang.csv
done