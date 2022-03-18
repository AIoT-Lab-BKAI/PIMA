LABELS = ['drugname', 'other']
labels_weight = [0.99, 0.01]

# Text_embedding Config
text_embedding = 768
text_pretrained = True
text_trainable = True

# Graph
graph_embedding = 256

# Image Encoder Config
image_path = 'data/small-pills/'
depth = 3
size = 224
image_model_name = 'resnet50'

# Default pretrained image_model_name
image_pretrained = False
image_trainable = True
# image_pretrained_link = '/mnt/disk1/vaipe-thanhnt/EMED-Prescription-and-Pill-matching/test/cv-model/pills-resnet/logs/model_5.pth'
image_embedding = 2048
image_batch_size = 16

# For projection head; used for both image and graph encoder
projection_dim = 256
dropout = 0.2
