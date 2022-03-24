import glob
import torch
from tqdm import tqdm
from models.prescription_pill import PrescriptionPill
from utils.metrics import ContrastiveLoss, TripletLoss, MetricTracker
import wandb
import config as CFG
from utils.utils import build_loaders, creat_batch_triplet, creat_batch_triplet_random
from utils.option import option
import warnings
warnings.filterwarnings("ignore")


def train(model, train_loader, optimizer, matching_criterion, graph_criterion, epoch, drugname_f1_score):
    model.train()
    train_loss = []
    wandb.watch(model)
    with tqdm(train_loader, desc=f"Train Epoch {epoch}") as train_bar:
        # Loop through for each prescription
        for data in train_bar:
            data = data.cuda()
            optimizer.zero_grad()
            pre_loss = []

            image_features, sentences_graph_features, graph_extract = model(
                data)
            # get graph feature
            graph_loss = graph_criterion(graph_extract, data.y)

            text_embedding_drugname = sentences_graph_features[data.pills_label >= 0]
            text_embedding_labels = data.pills_label[data.pills_label >= 0]

            anchor, positive, negative = creat_batch_triplet(
                image_features, text_embedding_drugname, text_embedding_labels, data.pills_images_labels)

            # anchor, positive, negative = creat_batch_triplet_random(
            #     image_features, sentences_graph_features, data.pills_label, data.pills_images_labels, 0.5)

            loss = matching_criterion(anchor, positive, negative) + (1 - drugname_f1_score) * graph_loss
            loss.backward()
            optimizer.step()
            pre_loss.append(loss.item())

            train_loss.append(sum(pre_loss) / len(pre_loss))
            train_bar.set_postfix(loss=train_loss[-1])

    return sum(train_loss) / len(train_loss)


def val(model, val_loader):
    """
    Summary: Test with all text embedding
    """
    metric = MetricTracker(labels=CFG.LABELS)
    model.eval()
    matching_acc = []
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation"):
            data = data.cuda()

            correct = []
            image_features, sentences_graph_features, graph_extract = model(
                data)
                
            # For Graph
            graph_predict = graph_extract.data.max(1, keepdim=True)[1]
            metric.update(graph_predict, data.y.data.view_as(graph_predict))

            ######
            # text_embedding_drugname = sentences_graph_features[data.pills_label >= 0]
            # text_embedding_labels = data.pills_label[data.pills_label >= 0]
            # similarity = image_features @ text_embedding_drugname.t()
            # _, predicted = torch.max(similarity, 1)
            # mapping_predicted = text_embedding_labels[predicted]
            ####

            # For Matching
            similarity = image_features @ sentences_graph_features.t()
            _, predicted = torch.max(similarity, 1)
            mapping_predicted = data.pills_label[predicted]

            correct.append(mapping_predicted.eq(
                data.pills_images_labels).sum().item() / len(data.pills_images_labels))

            matching_acc.append(sum(correct) / len(correct))

    final_accuracy = sum(matching_acc) / len(matching_acc)

    gcn_report = metric.compute(mode=True)
    drugname_f1_score = gcn_report['drugname']['f1-score']
    print(metric.compute())

    return final_accuracy, drugname_f1_score


def main(args):
    print("CUDA status: ", args.cuda)
    torch.cuda.manual_seed_all(args.seed)

    print(">>>> Preparing data...")
    train_files = glob.glob(args.train_folder + "*.json")
    # get 10% of train_files
    train_files = train_files[:int(len(train_files) * 0.1)]

    train_loader = build_loaders(
        train_files, mode="train", batch_size=args.train_batch_size)

    val_files = glob.glob(args.val_folder + "*.json")
    val_loader = build_loaders(
        val_files, mode="test", batch_size=args.val_batch_size)

    # Print data information
    print("Train files: ", len(train_files))
    print("Val files: ", len(val_files))

    print(">>>> Preparing model...")
    model = PrescriptionPill(args).cuda()

    print(">>>> Preparing optimizer...")
    if args.matching_criterion == "ContrastiveLoss":
        matching_criterion = ContrastiveLoss()
    elif args.matching_criterion == "TripletLoss":
        matching_criterion = TripletLoss()

    class_weights = torch.FloatTensor(CFG.labels_weight).cuda()
    graph_criterion = torch.nn.NLLLoss(weight=class_weights)

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_accuracy = 0
    drugname_f1_score = 0
    print(">>>> Training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, matching_criterion, graph_criterion, epoch, drugname_f1_score)
        print(">>>> Train Validation...")
        train_acc, drugname_f1_score = val(model, train_loader)
        print("Train accuracy: ", train_acc)

        # print(">>>> Test Validation...")
        # val_acc, drugname_f1_score = val(model, val_loader)
        # print("Val accuracy: ", val_acc)

        # wandb.log({"train_loss": train_loss,
        #           "train_acc": train_acc, "val_acc": val_acc})
        # if val_acc > best_accuracy:
        #     best_accuracy = val_acc
        #     print(">>>> Saving model...")
        #     torch.save(model.state_dict(), args.save_folder + "best_model.pth")


if __name__ == '__main__':
    parse_args = option()

    wandb.init(entity="aiotlab", project="VAIPE-Pills-Prescription-Matching", group="Graph", mode="disabled",
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
