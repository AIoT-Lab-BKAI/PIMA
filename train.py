import argparse
import glob
import logging
import os
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from data.data import PrescriptionPillData
from utils.metrics import MetricTracker, MatchingMetric
from utils import LABELS
from models.prescription_pill import PrescriptionPill
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

def train(model, train_loader, optimizer, criterion, matching_metric, lr_scheduler, epoch, log_writer):
    """[Train]

    Args:
        model ([type]): [description]
        train_loader ([type]): [description]
        optimizer ([type]): [description]
        criterion ([type]): [description]
        lr_scheduler ([type]): [description]
        epoch ([type]): [description]
        log_writer ([type]): [description]

    Returns:
        [type]: [description]
    """

    model.train()
    train_loss, extractLoss, pillPrescriptionLoss = 0., 0., 0.

    with tqdm(train_loader, desc=f"Train Epoch {epoch}") as train_bar:
        for batch_idx, data in enumerate(train_bar):
            if args.cuda:
                data = data.cuda()
            optimizer.zero_grad()
            
            graph_extract, image_embeddings, graph_embeddings = model(data)
            
            # Matching loss 
            loss_graph_images = matching_metric.compute_loss(image_embeddings, graph_embeddings, data.matching_label)

            # KIE Loss
            extract_loss = criterion(graph_extract, data.y) 

            loss = extract_loss + 1 * loss_graph_images
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss += loss
            extractLoss += extract_loss
            pillPrescriptionLoss += loss_graph_images


    train_loss /= len(train_loader)
    extractLoss /= len(train_loader)
    pillPrescriptionLoss /= len(train_loader)

    if log_writer:
        log_writer.add_scalar('train/loss', loss, epoch)
    return train_loss, extractLoss, pillPrescriptionLoss


def val(model, val_loader, criterion, matching_metric, epoch, metric, log_writer):
    """[summary]

    Args:
        model ([type]): [description]
        val_loader ([type]): [description]
        criterion ([type]): [description]
        epoch ([type]): [description]
        metric ([type]): [description]
        log_writer ([type]): [description]

    Returns:
        [type]: [description]
    """
    model.eval()
    val_loss = 0.
    matching_acc = []

    for data in tqdm(val_loader, desc="Validation"):
        if args.cuda:
            data = data.cuda()
        with torch.no_grad():
            graph_extract, image_embeddings, graph_embeddings = model(data)

            # Matching loss
            loss_graph_images = matching_metric.compute_loss(image_embeddings, graph_embeddings, data.matching_label)
            # KIE Loss
            extract_loss = criterion(graph_extract, data.y)

            loss = extract_loss + 1 * loss_graph_images
            val_loss += loss.item()

            # get the index of the max log-probability (KIE)
            pred = graph_extract.data.max(1, keepdim=True)[1]
            metric.update(pred, data.y.data.view_as(pred))

            # Matching accuracy
            matching_acc.append(matching_metric.accuracy(image_embeddings, graph_embeddings, data.matching_label))

    
    val_loss /= len(val_loader)
    print(f"Classification Report:")
    print(metric.compute())

    print(f"Matching Accuracy:", sum(matching_acc) / len(matching_acc))

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss, epoch)
    metric.reset()
    return val_loss

def main(args):
    """[Main]
    Args:
        args ([type]): [description]
    """
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print("CUDA status: ", args.cuda)
    
    torch.manual_seed(args.seed)
    if args.cuda:
        # Set cuda device 
        # device = "cuda"
        torch.device("cuda")
        # torch.cuda.set_device()
        torch.cuda.manual_seed(args.seed)

    # Load data
    train_files = glob.glob(args.train_folder + "*.json")
    train_loader = build_loaders(train_files, mode="train")

    val_files = glob.glob(args.val_folder + "*.json")
    val_loader = build_loaders(val_files, mode="test")

    # Print data information
    print("Train files: ", len(train_files))
    print("Val files: ", len(val_files))


    model = PrescriptionPill()
    if args.cuda:
        model.cuda()
    
    class_weights = torch.FloatTensor([2.0, 1.5, 1.0, 1.0, 1.0, 0.2]).cuda()
    criterion = torch.nn.NLLLoss(weight=class_weights)
    matching_metric = MatchingMetric(temperature=CFG.temperature)

    # Define optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    t_total = len(train_loader) * args.epochs

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=t_total)

    metric = MetricTracker(labels=LABELS)
    best_loss = -1
    log_writer = SummaryWriter(args.log_dir)

    for epoch in range(1, args.epochs + 1):
        train_loss, extractLoss, pillPrescriptionLoss = train(model, train_loader, optimizer, criterion, matching_metric, lr_scheduler, epoch, log_writer)
        val_loss = val(model, val_loader, criterion, matching_metric, epoch, metric, log_writer)

        print('Train Epoch: {} \nTrain Loss: {:.6f} \tExtract Loss: {:.6f} \tPill Prescription Loss: {:.6f} \nValidation Loss: {:.6f} \t'.format(
            epoch, train_loss, extractLoss, pillPrescriptionLoss, val_loss))

        if args.save_interval > 0:
            if val_loss < best_loss or best_loss < 0:
                best_loss = val_loss
                print(f"Saving best model, loss: {best_loss}")
                torch.save(model, os.path.join(
                    args.save_folder, "model_best.pth"))
                continue
            if epoch % args.save_interval == 0:
                print(f"Saving at epoch: {epoch}")
                torch.save(model, os.path.join(
                    args.save_folder, f"model_{epoch}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch BERT-GCN')
    
    parser.add_argument('--batch-size', type=int, default=2, metavar='N')
    parser.add_argument('--val-batch-size', type=int, default=2, metavar='N')

    parser.add_argument('--train-folder', type=str,
                        default="data/prescriptions/train/",
                        help='training folder path')
    parser.add_argument('--val-folder', type=str,
                        default="data/prescriptions/test/",
                        help='validation folder path')

    parser.add_argument('--log-dir', type=str,
                        default="logs/runs/",
                        help='TensorBoard folder path')    

    parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
