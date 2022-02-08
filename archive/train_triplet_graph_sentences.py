import argparse
import glob
import torch
from torch import nn
from torch_geometric.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from data.data import PrescriptionPillData
from models.prescription_pill import PrescriptionPill
import config as CFG
from utils.metrics import TripletLoss


def build_loaders(files, mode="train"):
    dataset = PrescriptionPillData(files, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=True)

    return dataloader


def creat_batch_triplet_random(image_embedding, graph_embedding_pills, graph_embedding_pills_labels, labels, random_percent=0.2):
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
            random_negative = graph_embedding_pills[negative_idx]
            random_negative = random_negative[torch.randperm(len(random_negative))]
            random_negative = random_negative[:int(random_percent * len(random_negative))]
            negative = torch.cat((negative, random_negative.unsqueeze(0)))
    
    return anchor, positive, negative

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


def train(model, train_loader, optimizer, matching_criterion, lr_scheduler, epoch):
    model.train()

    train_loss = []
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

                anchor, positive, negative = creat_batch_triplet_random(image_embedding, text_embedding, data.pills_label, labels, random_percent=0.3)                

                loss = matching_criterion(anchor, positive, negative)
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                pre_loss.append(loss.item())

            train_loss.append(sum(pre_loss) / len(pre_loss))
            train_bar.set_postfix(loss=train_loss[-1])
            
    print("Train_loss: ", sum(train_loss) / len(train_loss))



def val(model, val_loader):
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

                image_embedding, text_embedding = model(data, images)
                
                similarity = image_embedding @ text_embedding.t()
                _, predicted = torch.max(similarity, 1)
                mapping_predicted = data.pills_label[predicted]

                correct.append(mapping_predicted.eq(labels).sum().item() / len(labels))
            matching_acc.append(sum(correct) / len(correct))
    print("Val_acc: ", sum(matching_acc) / len(matching_acc))


def main(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
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

    # TODO:
    matching_criterion = TripletLoss()

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=5e-4)
    t_total = len(train_loader) * args.epochs

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=t_total)


    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer,
              matching_criterion, lr_scheduler, epoch)

        print(">>>> Train Validation...")
        val(model, train_loader)

        print(">>>> Test Validation...")
        val(model, val_loader)


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
    parser.add_argument('--log-dir', type=str,
                        default="logs/runs/",
                        help='TensorBoard folder path')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--num-warmup-steps', type=float, default=1000, metavar='N',
                        help='numbers warmup steps (default: 1000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                        help='how many epoches to wait before saving model')
    parser.add_argument('--save-folder', type=str, default="logs/saved", metavar='N',
                        help='how many epoches to wait before saving model')

    args = parser.parse_args()

    main(args)
