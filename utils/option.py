import argparse
import torch


def option():
    """
    This function is used to get the user's input.
    """

    parser = argparse.ArgumentParser()

    # Wandb
    parser.add_argument('--run-name', type=str, default='')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())

    # Data Init
    parser.add_argument('--data-folder', type=str, default='data/')
    parser.add_argument('--train-folder', type=str,
                        default="prescriptions/train/")
    parser.add_argument('--val-folder', type=str,
                        default="prescriptions/test/")

    parser.add_argument('--image-path', type=str, default="pills/")
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--image-size', type=int, default=224)

    parser.add_argument('--train-batch-size', type=int, default=1)
    parser.add_argument('--val-batch-size', type=int, default=1)

    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)

    # Model
    # Text Model
    parser.add_argument('--text-model-name', type=str,
                        default="sentence-transformers/paraphrase-mpnet-base-v2")
    parser.add_argument('--text-embedding', type=int, default=768)
    parser.add_argument('--text-encoder-model', type=str,
                        default="bert-base-cased")
    parser.add_argument('--text-pretrained', type=bool, default=False)
    parser.add_argument('--text-trainable', type=bool, default=False)

    # Graph Model
    parser.add_argument('--graph-embedding', type=int, default=256)

    # Image Model
    parser.add_argument('--image-model-name', type=str, default="resnet50")
    parser.add_argument('--image-pretrained', type=bool, default=False)
    parser.add_argument('--image-trainable', type=bool, default=False)
    parser.add_argument('--image-embedding', type=int, default=2048)

    # for projection head; used for both image and graph encoder
    parser.add_argument('--num-projection-layers', type=int, default=1)
    parser.add_argument('--projection-dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.2)

    # matching
    parser.add_argument('--matching-criterion', type=str,
                        default="ContrastiveLoss")
    parser.add_argument('--negative-ratio', type=float, default=None)

    # Model Save
    parser.add_argument('--save-model', type=bool, default=False)
    parser.add_argument('--save-folder', type=str, default="logs/saved/")
    parser.add_argument('--run-group', type=str, default="")

    return parser.parse_args()
