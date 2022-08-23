CUDA_VISIBLE_DEVICES=0 python3 -u train.py --run-name="PP-02-Graph-Mobile-NonPretrain-SBertMulti" --run-group="VAIPE-PP_02" --image-model-name="mobilenet_v3_small" --image-embedding=1000 --image-trainable=True --image-pretrained=False --matching-criterion="ContrastiveLoss" --text-model-name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" --text-embedding=384 --train-batch-size=4 --val-batch-size=1 --data-folder="/mnt/disk1/vaipe-thanhnt/EMED-Prescription-and-Pill-matching/data/VAIPE-PP_02/"
 