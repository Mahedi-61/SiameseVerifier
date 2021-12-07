import os
import torch 

# directories
root_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(root_dir, "model")

train_dir = os.path.join(root_dir, "data", "omniglot", "images_background")
test_dir = os.path.join(root_dir, "data", "omniglot", "images_evaluation")


# model hyperparameters
batch_size = 512
learning_rate = 0.0005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 300
img_size = 96

# training parameters
multi_gpus = True 
load_model = False 
save_model = False
way = 20
times = 400 