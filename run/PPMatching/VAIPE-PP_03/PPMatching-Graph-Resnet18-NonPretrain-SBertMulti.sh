python3 -u train.py --run-name="PPMatching-Graph-Resnet18-NonPretrain-SBertMulti" --image-model-name="resnet18" --image-embedding=512 --image-trainable=True --image-pretrained=False --matching-criterion="ContrastiveLoss" --text-model-name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" --text-embedding=384 --train-batch-size=4 --val-batch-size=1 --data-folder="/mnt/disk2/thanhnt/ThanhNT-Data/2904_VAIPE-Matching/VAIPE-PP_03/data/"
