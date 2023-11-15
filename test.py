import os
import torch.utils.data
from engine import evaluate
import utils
import train

path = os.getcwd()
dataset = 'Drinks'

path_to_data = os.path.join(path, dataset, 'drinks')
dataset_val = train.DrinksDataset(path_to_data, os.path.join(path_to_data, 'labels_test.csv'), transforms=train.get_transform(train=False))
data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1, collate_fn=utils.collate_fn)
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda:0')
# 3 class (number of classname) + 1 class (background)
num_classes = 4
model = train.get_instance_segmentation_model(num_classes)
model.load_state_dict(torch.load(os.path.join(path, 'model_weights.pth')))
model.to(device)
evaluate(model, data_loader_val, device=device)
