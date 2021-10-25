
cuda = 2

# graph 
graph_embedding = 256

# image size
image_path = 'data/pills'
depth = 3 # RGB Image
size = 224
model_name = 'resnet50'
pretrained = False
trainable = False
image_embedding = 2048

# for projection head; used for both image and graph encoder
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1

temperature = 1.0
