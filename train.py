import argparse
import glob
import logging
import os
from networkx.algorithms import similarity
from numpy import positive
import torch
from torch import nn
from torch_geometric.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from data.data import PrescriptionPillData
from utils.metrics import MetricTracker, MatchingMetric
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


def train(model, train_loader, optimizer, criterion, matching_criterion, lr_scheduler, epoch, log_writer):
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

    train_loss = []
    with tqdm(train_loader, desc=f"Train Epoch {epoch}") as train_bar:
        # Loop through for each prescription 
        for data in train_bar:
            if args.cuda:
                data = data.cuda()
            optimizer.zero_grad()

            pre_loss = []
            pills_loader = torch.utils.data.DataLoader(data.pills_from_folder[0], batch_size=1, shuffle=True, num_workers=4)
            for images, labels in pills_loader:
                if args.cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                
                image_embedding, graph_embedding = model(data, images)

                positive_idx = data.pills_label.eq(labels).nonzero().squeeze()
                negative_idx = data.pills_label.ne(labels).nonzero().squeeze()

                anchor = image_embedding                 
                positive = graph_embedding[positive_idx]
                positive = positive.unsqueeze(0)
                negative = graph_embedding[negative_idx]

                loss = matching_criterion(anchor, positive, negative)
                loss.backward()
                optimizer.step()

                pre_loss.append(loss.item())
                # break
            
            lr_scheduler.step()
            train_loss.append(sum(pre_loss) / len(pre_loss))
            # break
        
    print("Train_loss: ", sum(train_loss) / len(train_loss))



def val(model, val_loader, criterion, matching_criterion, epoch, metric, log_writer):
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
    
    matching_acc = []
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation"):
            if args.cuda:
                data = data.cuda()
            
            correct = []
            pills_loader = torch.utils.data.DataLoader(data.pills_from_folder[0], batch_size=1, shuffle=True, num_workers=4)
            for images, labels in pills_loader:
                if args.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                image_embedding, graph_embedding = model(data, images)

                # consine similarity image_embedding, graph_embedding
                similarity = nn.functional.cosine_similarity(image_embedding, graph_embedding, dim=1)
                idx = similarity.argmax()
                predict_label = data.pills_label[idx].item()
                correct.append(predict_label == labels.item())
                # break
            # break
            matching_acc.append(sum(correct) / len(correct))
    print("Val_acc: ", sum(matching_acc) / len(matching_acc))



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


    class_weights = torch.FloatTensor(CFG.labels_weight).cuda()
    criterion = torch.nn.NLLLoss(weight=class_weights)

    # TODO:
    matching_criterion = nn.TripletMarginWithDistanceLoss(distance_function = nn.CosineSimilarity(dim=1, eps=1e-6))

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    t_total = len(train_loader) * args.epochs

    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=t_total)

    # Tracker Graph
    metric = MetricTracker(labels=CFG.LABELS)
    best_loss = -1
    log_writer = SummaryWriter(args.log_dir)

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, criterion,
              matching_criterion, lr_scheduler, epoch, log_writer)

        print(">>>> Train Validation...")
        val(model, train_loader, criterion, matching_criterion, epoch, metric, log_writer)

        print(">>>> Test Validation...")
        val(model, val_loader, criterion, matching_criterion, epoch, metric, log_writer)

        # train_loss, extractLoss, pillPrescriptionLoss = train(
        #     model, train_loader, optimizer, criterion, matching_criterion, lr_scheduler, epoch, log_writer)
        # print("Train Validation: ")
        # train_loss = val(model, train_loader, criterion,
        #                  matching_criterion, epoch, metric, log_writer)

        # print("Test Validation")
        # val_loss = val(model, val_loader, criterion,
        #                matching_criterion, epoch, metric, log_writer)

        # print('Train Epoch: {} \nTrain Loss: {:.6f} \tExtract Loss: {:.6f} \tPill Prescription Loss: {:.6f} \nValidation Loss: {:.6f} \t'.format(
        #     epoch, train_loss, extractLoss, pillPrescriptionLoss, val_loss))

        # if args.save_interval > 0:
        #     if val_loss < best_loss or best_loss < 0:
        #         best_loss = val_loss
        #         print(f"Saving best model, loss: {best_loss}")
        #         torch.save(model, os.path.join(
        #             args.save_folder, "model_best.pth"))
        #         continue
        #     if epoch % args.save_interval == 0:
        #         print(f"Saving at epoch: {epoch}")
        #         torch.save(model, os.path.join(
        #             args.save_folder, f"model_{epoch}.pth"))


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
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
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
