import argparse
import glob
import torch
from torch import nn
from torch_geometric.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from data.data import PrescriptionPillData
from models.prescription_pill import PrescriptionPill
from utils.metrics import ContrastiveLoss, MetricTracker
import wandb
import config as CFG


def build_loaders(files, mode="train"):
    """[Build Loader]

    Args:
        files ([type]): [description]

    Returns:
        [DataLoader]
    """
    dataset = PrescriptionPillData(files, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True)

    return dataloader


def creat_batch_triplet(image_embedding, graph_embedding_pills, graph_embedding_pills_labels, labels):
    """[summary]

    Args:
        image_embedding ([type]): [description]
        graph_embedding ([type]): [description]
        labels ([type]): [description]

    Returns:
        [type]: [description]
    """
    anchor, positive, negative = torch.tensor([]).cuda(
    ), torch.tensor([]).cuda(), torch.tensor([]).cuda()

    for idx, label in enumerate(labels):
        positive_idx = graph_embedding_pills_labels.eq(label)
        negative_idx = graph_embedding_pills_labels.ne(label)

        anchor = torch.cat(
            (anchor, image_embedding[idx].unsqueeze(0).unsqueeze(0)))
        positive = torch.cat(
            (positive, graph_embedding_pills[positive_idx].unsqueeze(0)))

        if sum(negative_idx) == 0:           
            negative = torch.cat((negative, torch.zeros_like(image_embedding[idx]).unsqueeze(0).unsqueeze(0)))
        else:
            negative = torch.cat((negative, graph_embedding_pills[negative_idx].unsqueeze(0)))
    
    return anchor, positive, negative


def train(model, train_loader, optimizer, matching_criterion, graph_criterion, lr_scheduler, epoch):
    model.train()
    train_loss = []
    wandb.watch(model)
    with tqdm(train_loader, desc=f"Train Epoch {epoch}") as train_bar:
        # Loop through for each prescription
        for data in train_bar:
            if args.cuda:
                data = data.cuda()
            optimizer.zero_grad()

            pre_loss = []
            pills_loader = torch.utils.data.DataLoader(
                data.pills_from_folder[0], batch_size=16, shuffle=True, num_workers=4)
            
            for images, labels in pills_loader:
                if args.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                image_embedding, text_embedding = model(data, images)
                # get graph feature
                graph_feature = model.forward_graph(data)
                graph_loss = graph_criterion(graph_feature, data.y) 

                text_embedding_drugname = text_embedding[data.pills_label >= 0]
                text_embedding_labels = data.pills_label[data.pills_label >= 0]
                anchor, positive, negative = creat_batch_triplet(image_embedding, text_embedding_drugname, text_embedding_labels, labels)                  

                loss = matching_criterion(anchor, positive, negative) + graph_loss
                loss.backward()

                optimizer.step()
                pre_loss.append(loss.item())
            lr_scheduler.step()

            train_loss.append(sum(pre_loss) / len(pre_loss))
            train_bar.set_postfix(loss=train_loss[-1])

    return sum(train_loss) / len(train_loss)


def val(model, val_loader, mode="train"):
    """
    Summary: Test with all text embedding
    """
    metric = MetricTracker(labels=CFG.LABELS)
    model.eval()
    matching_acc = []
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation"):
            if args.cuda:
                data = data.cuda()

            correct = []
            pills_loader = torch.utils.data.DataLoader(
                data.pills_from_folder[0], batch_size=16, shuffle=True, num_workers=4)
            for images, labels in pills_loader:
                if args.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                # For Graph 
                graph_feature = model.forward_graph(data)
                graph_predict = graph_feature.data.max(1, keepdim=True)[1]
                metric.update(graph_predict, data.y.data.view_as(graph_predict))

                # For Matching
                image_embedding, text_embedding = model(data, images)
                
                # Graph support
                graph_text_embedding = text_embedding[graph_predict.squeeze(1) == CFG.drugname_label]
                graph_text_embedding_label = data.pills_label[graph_predict.squeeze(1) == CFG.drugname_label]

                similarity = image_embedding @ graph_text_embedding.t()
                _, predicted = torch.max(similarity, 1)
                mapping_predicted = graph_text_embedding_label[predicted]

                correct.append(mapping_predicted.eq(labels).sum().item() / len(labels))
            matching_acc.append(sum(correct) / len(correct))

    final_accuracy = sum(matching_acc) / len(matching_acc)

    print(f"Classification Report:")
    print(metric.compute())    
    return final_accuracy

def main(args):
    print("CUDA status: ", args.cuda)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.device("cuda")
        torch.cuda.manual_seed(args.seed)

    print(">>>> Preparing data...")
    # Load data
    train_files = glob.glob(args.train_folder + "*.json")
    train_loader = build_loaders(train_files, mode="train")

    val_files = glob.glob(args.val_folder + "*.json")
    val_loader = build_loaders(val_files, mode="test")

    # Print data information
    print("Train files: ", len(train_files))
    print("Val files: ", len(val_files))

    print(">>>> Preparing model...")
    model = PrescriptionPill()
    if args.cuda:
        model.cuda()

    print(">>>> Preparing optimizer...")
    matching_criterion = ContrastiveLoss()
    class_weights = torch.FloatTensor(CFG.labels_weight).cuda()
    graph_criterion = torch.nn.NLLLoss(weight=class_weights)

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=5e-4)
    t_total = len(train_loader) * args.epochs

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=t_total)

    best_accuracy = 0
    print(">>>> Training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer,
              matching_criterion, graph_criterion, lr_scheduler, epoch)

        print(">>>> Train Validation...")
        train_acc = val(model, train_loader)
        print("Train accuracy: ", train_acc)
        
        print(">>>> Test Validation...")
        val_acc = val(model, val_loader, mode="Val")
        print("Val accuracy: ", val_acc)

        wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
        # if val_acc > best_accuracy:
        #     best_accuracy = val_acc
        #     print(">>>> Saving model...")
        #     torch.save(model.state_dict(), args.save_folder + "best_model.pth")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch BERT-GCN')

    parser.add_argument('--batch-size', type=int, default=1, metavar='N')
    parser.add_argument('--val-batch-size', type=int, default=1, metavar='N')
    parser.add_argument('--train-folder', type=str,
                        default="data/prescriptions/train/",
                        help='training folder path')
    parser.add_argument('--val-folder', type=str,
                        default="data/prescriptions/test/",
                        help='validation folder path')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--num-warmup-steps', type=float, default=1000, metavar='N',
                        help='numbers warmup steps (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--save-folder', type=str, default="logs/saved/", metavar='N',
                        help='how many epoches to wait before saving model')

    parse_args = parser.parse_args()

    wandb.init(project="vaipe-pills-prescription-matching", entity="thanhhff",
    config = {
        "batch_size": parse_args.batch_size,
        "val_batch_size": parse_args.val_batch_size,
        "epochs": parse_args.epochs,
        "lr": parse_args.lr,
        "num_warmup_steps": parse_args.num_warmup_steps,
        "train_folder": parse_args.train_folder,
        "val_folder": parse_args.val_folder,
        "save_folder": parse_args.save_folder,
        "cuda": torch.cuda.is_available(),
        "seed": parse_args.seed
    })
    args = wandb.config 
    main(args)
    wandb.finish()
