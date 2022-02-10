import glob
import torch
from tqdm import tqdm
from models.image_text import ImageTextMatching
from utils.metrics import ContrastiveLoss, TripletLoss
import wandb
from utils.utils import build_loaders, creat_batch_triplet
from utils.option import option
from transformers import get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings("ignore")


def train(model, train_loader, optimizer, matching_criterion, epoch, lr_scheduler):
    model.train()
    train_loss = []
    wandb.watch(model)
    with tqdm(train_loader, desc=f"Train Epoch {epoch}") as train_bar:
        # Loop through for each prescription
        for data in train_bar:
            data = data.cuda()

            optimizer.zero_grad()

            pre_loss = []
            pills_loader = torch.utils.data.DataLoader(
                data.pills_from_folder[0], batch_size=8, shuffle=True, num_workers=4)

            for images, labels in pills_loader:
                images = images.cuda()
                labels = labels.cuda()

                image_features, sentences_graph_features = model(data, images)

                text_embedding_drugname = sentences_graph_features[data.pills_label >= 0]
                text_embedding_labels = data.pills_label[data.pills_label >= 0]
                anchor, positive, negative = creat_batch_triplet(
                    image_features, text_embedding_drugname, text_embedding_labels, labels)

                loss = matching_criterion(anchor, positive, negative)
                loss.backward()

                optimizer.step()
                pre_loss.append(loss.item())
                lr_scheduler.step()

            train_loss.append(sum(pre_loss) / len(pre_loss))
            train_bar.set_postfix(loss=train_loss[-1])

    return sum(train_loss) / len(train_loss)


def val(model, val_loader):
    """
    Summary: Test with all text embedding
    """
    model.eval()
    matching_acc = []
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation"):
            data = data.cuda()

            correct = []
            pills_loader = torch.utils.data.DataLoader(
                data.pills_from_folder[0], batch_size=8, shuffle=True, num_workers=4)
            for images, labels in pills_loader:
                images = images.cuda()
                labels = labels.cuda()

                image_features, sentences_graph_features = model(data, images)

                # For Matching
                similarity = image_features @ sentences_graph_features.t()
                _, predicted = torch.max(similarity, 1)
                mapping_predicted = data.pills_label[predicted]

                correct.append(mapping_predicted.eq(
                    labels).sum().item() / len(labels))
            matching_acc.append(sum(correct) / len(correct))

    final_accuracy = sum(matching_acc) / len(matching_acc)
    return final_accuracy


def main(args):
    print("CUDA status: ", args.cuda)
    torch.cuda.manual_seed_all(args.seed)

    print(">>>> Preparing data...")
    # Load data
    train_files = glob.glob(args.train_folder + "*.json")
    train_loader = build_loaders(
        train_files, mode="train", batch_size=args.train_batch_size)

    val_files = glob.glob(args.val_folder + "*.json")
    val_loader = build_loaders(
        val_files, mode="test", batch_size=args.val_batch_size)

    # Print data information
    print("Train files: ", len(train_files))
    print("Val files: ", len(val_files))

    print(">>>> Preparing model...")
    model = ImageTextMatching().cuda()

    print(">>>> Preparing optimizer...")
    if args.matching_criterion == "ContrastiveLoss":
        matching_criterion = ContrastiveLoss()
    elif args.matching_criterion == "TripletLoss":
        matching_criterion = TripletLoss()

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=5e-4)
    t_total = len(train_loader) * args.epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=1000, num_training_steps=t_total)

    best_accuracy = 0
    print(">>>> Training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer,
                           matching_criterion, epoch, lr_scheduler)

        print(">>>> Train Validation...")
        train_acc = val(model, train_loader)
        print("Train accuracy: ", train_acc)

        print(">>>> Test Validation...")
        val_acc = val(model, val_loader)
        print("Val accuracy: ", val_acc)

        wandb.log({"train_loss": train_loss,
                  "train_acc": train_acc, "val_acc": val_acc})
        # if val_acc > best_accuracy:
        #     best_accuracy = val_acc
        #     print(">>>> Saving model...")
        #     torch.save(model.state_dict(), args.save_folder + "best_model.pth")


if __name__ == '__main__':
    parse_args = option()

    wandb.init(entity="aiotlab", project="VAIPE-Pills-Prescription-Matching", group="Non-Graph",
               config={
                   "train_batch_size": parse_args.train_batch_size,
                   "val_batch_size": parse_args.val_batch_size,
                   "epochs": parse_args.epochs,
                   "lr": parse_args.lr,
                   "seed": parse_args.seed
               })
    args = wandb.config

    wandb.define_metric("val_acc", summary="max")
    main(parse_args)
    wandb.finish()
