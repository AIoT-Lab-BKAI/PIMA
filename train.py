import glob
import torch
from tqdm import tqdm
from models.prescription_pill import PrescriptionPill
from utils.metrics import ContrastiveLoss, TripletLoss
import wandb
from utils.utils import build_loaders, calculate_matching_loss
from utils.option import option
import warnings
import config as CFG
warnings.filterwarnings("ignore")


def train(model, train_loader, optimizer, matching_criterion, graph_criterion, epoch):
    model.train()
    train_loss = []
    wandb.watch(model)
    with tqdm(train_loader, desc=f"Train Epoch {epoch}") as train_bar:
        # Loop through for each prescription
        for data in train_bar:
            data = data.cuda()
            optimizer.zero_grad()
            pre_loss = []

            image_aggregation, sentences_projection, graph_extract = model(data)
            _, max_graph_extract = torch.max(graph_extract, 1)

            # Create for Image matching Drugname
            sentences_embedding_drugname = sentences_projection[max_graph_extract == 0]
            sentences_labels_drugname = data.pills_label[max_graph_extract == 0]
            
            matching_loss = calculate_matching_loss(image_aggregation, sentences_embedding_drugname, sentences_labels_drugname, data.pills_images_labels, matching_criterion)

            # Create for Image matching Graph
            graph_loss = graph_criterion(graph_extract, data.y)
            loss = matching_loss + graph_loss

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
    model.eval()
    matching_acc = []
    
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation"):
            data = data.cuda()

            correct = []
            image_aggregation, sentences_projection, graph_extract = model(data)

            # For Matching
            similarity_text_matching = image_aggregation @ sentences_projection.t() 
            similarity_text_matching = torch.nn.functional.softmax(similarity_text_matching, dim=1)
            similarity = torch.where(similarity_text_matching > 0.8, similarity_text_matching, torch.zeros_like(similarity_text_matching))
            
            _, predicted = torch.max(similarity, 1)
            mapping_predicted = data.pills_label[predicted]

            correct.append(mapping_predicted.eq(
                data.pills_images_labels).sum().item() / len(data.pills_images_labels))

            matching_acc.append(sum(correct) / len(correct))

    final_accuracy = sum(matching_acc) / len(matching_acc)

    return final_accuracy


def main(args):
    print("CUDA status: ", args.cuda)
    torch.cuda.manual_seed_all(args.seed)

    print(">>>> Preparing data...")
    train_files = glob.glob(args.data_folder + args.train_folder + "*.json")
    val_files = glob.glob(args.data_folder + args.val_folder + "*.json")

    train_loader = build_loaders(
        train_files, mode="train", batch_size=args.train_batch_size, args=args)
    # train_val_loader = build_loaders(
    #     train_files, mode="train", batch_size=args.val_batch_size, args=args)
    val_loader = build_loaders(
        val_files, mode="test", batch_size=args.val_batch_size, args=args)

    # Print data information
    print("Train files: ", len(train_files))
    print("Val files: ", len(val_files))

    print(">>>> Preparing model...")
    model = PrescriptionPill(args).cuda()

    print(">>>> Preparing optimizer...")
    matching_criterion = ContrastiveLoss()
    class_weights = torch.FloatTensor(CFG.labels_weight).cuda()
    graph_criterion = torch.nn.NLLLoss(weight=class_weights)

    # Define optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=5e-4)

    best_accuracy = 0
    print(">>>> Training...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, matching_criterion, graph_criterion, epoch)
        print(">>>> Train Validation...")
        train_val_acc = 0 #val(model, train_val_loader)
        print("Train accuracy: ", train_val_acc)

        print(">>>> Test Validation...")
        val_acc = val(model, val_loader)
        print("Val accuracy: ", val_acc)
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            model_dir = "/mnt/disk1/vaipe-thanhnt/EMED-Prescription-and-Pill-matching/logs/weights/"
            model_name = "model_best.pth"
            model_path = model_dir + model_name
            torch.save(model.state_dict(), model_path)

        wandb.log({"train_loss": train_loss,
                  "train_acc": train_val_acc, "val_acc": val_acc})

if __name__ == '__main__':
    parse_args = option()

    wandb.init(entity="aiotlab", project="VAIPE-Pills-Prescription-Matching", group="Public", name=parse_args.run_name, #mode="disabled",
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
