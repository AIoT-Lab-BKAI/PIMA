python3 -u train.py --run-name="Resnet18-20%-TrainableText" --image-model-name="resnet18" --image-embedding=512 --image-trainable=True --image-pretrained=False --text-trainable=True --matching-criterion="ContrastiveLoss" 