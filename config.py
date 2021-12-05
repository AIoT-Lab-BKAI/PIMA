# cuda = 2

LABELS = ['drugname', 'diagnose', 'usage', 'quantity', 'date', 'other']
labels_weight = [1.5, 1.5, 1.0, 1.0, 1.0, 0.2]
# graph
graph_embedding = 256

### Image Encoder Config
image_path = 'data/pills/'
depth = 3  # RGB Image
size = 224
image_model_name = 'resnet50'
# Default pretrained image_model_name
image_pretrained = False 
image_trainable = True
image_pretrained_link = '/mnt/disk1/vaipe-thanhnt/EMED-Prescription-and-Pill-matching/pills-resnet/logs/model_5.pth'
image_embedding = 2048
image_batch_size = 16

# for projection head; used for both image and graph encoder
num_projection_layers = 1
projection_dim = 256
dropout = 0.2

temperature = 1.0
