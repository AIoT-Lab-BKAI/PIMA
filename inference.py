import glob
import torch
from tqdm import tqdm
from models.prescription_pill import PrescriptionPill
from data_inference.data_inference import PrescriptionPillData
from torch_geometric.data import DataLoader
from utils.option import option
from torch import nn

def build_loaders(files, batch_size=1, args=None):
    dataset = PrescriptionPillData(files, args)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True)
    return dataloader

def inference(args):
    
    inference_file = glob.glob("/mnt/disk1/vaipe-thanhnt/EMED-Prescription-and-Pill-matching/data_inference/data/pres/" + "*.json")    
    inference_loaders = build_loaders(inference_file, batch_size=args.val_batch_size, args=args)
    
    model = PrescriptionPill(args).cuda()
    model.load_state_dict(torch.load("/mnt/disk1/vaipe-thanhnt/EMED-Prescription-and-Pill-matching/logs/weights/model_best.pth"))
    model.eval()
    
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    
    for data in inference_loaders:
        data = data.cuda()
        
        print(data.pills_labels)
        image_aggregation, sentences_projection, graph_extract = model(data)
        for image in image_aggregation:
            similarity = cos(image, sentences_projection)       
            similarity = torch.where(similarity > 0.8, similarity, torch.zeros_like(similarity))
                 
            _, predicted = torch.max(similarity, 0)
            if predicted.item() == 0:
                print("Predicted: No match")
            else:
                print("Predicted: ", data.text[0][predicted.item()])
            
if __name__ == '__main__':
    parse_args = option()
    inference(parse_args)
