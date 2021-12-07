import torch
from torch.utils.data import Dataset
import config 
from torchvision import transforms 
import numpy as np
import os, random
from PIL import Image

class OmniglotTrain(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path 
        self.img_lists, self.num_classes = self.loadingData()

        self.trans = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.RandomAffine(15),
            transforms.ColorJitter(brightness=0.2),
            transforms.RandomRotation(degrees=(0, 15), fill=(255, )),
            transforms.ToTensor()
        ])

    def loadingData(self):
        img_data = {}
        indx = 0

        for lang in os.listdir(self.data_path):
            for alpha_set in os.listdir(os.path.join(self.data_path, lang)):
                alpha_set_dir = os.path.join(self.data_path, lang, alpha_set)

                img_data[indx] = [Image.open(os.path.join(alpha_set_dir, alpha_img)).
                            convert("L") for alpha_img in os.listdir(alpha_set_dir)]
                indx += 1

        return img_data, indx


    def __len__(self):
        return 10000


    def __getitem__(self, index):

        # get image from same class (genuine pair)
        if index % 2 == 0:
            label = 1.0
            class_indx = random.randint(0, self.num_classes-1)
            img1 = random.choice(self.img_lists[class_indx])
            img2 = random.choice(self.img_lists[class_indx])

        else: #imposter pair
            label = 0.0
            class_indx1 = random.randint(0, self.num_classes-1)
            class_indx2 = random.randint(0, self.num_classes-1)

            while class_indx1 == class_indx2:
                class_indx1 = random.randint(0, self.num_classes-1) 
            else: 
                img1 = random.choice(self.img_lists[class_indx1])
                img2 = random.choice(self.img_lists[class_indx2])

        img1 = self.trans(img1)
        img2 = self.trans(img2)
        return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))


class OmniglotTest(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.img_lists, self.num_classes = OmniglotTrain(self.data_path).loadingData()
        self.trans = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor()
        ])

        self.times = config.times
        self.way = config.way 


    def __len__(self):
        return self.way * self.times 


    def __getitem__(self, index):
        idx = index % self.way 
        if idx == 0: 
            label = 1.0
            class_indx = random.randint(0, self.num_classes-1)
            img1 = random.choice(self.img_lists[class_indx])
            img2 = random.choice(self.img_lists[class_indx])

        else:
            label = 0.0
            indx1 = random.randint(0, self.num_classes-1)
            indx2 = random.randint(0, self.num_classes-1)
            img1 = random.choice(self.img_lists[indx1])
            img2 = random.choice(self.img_lists[indx2])

            while (indx1 == indx2):
                indx2 = random.randint(0, self.num_classes-1)

        img1 = self.trans(img1)
        img2 = self.trans(img2)
        return img1, img2, torch.from_numpy(np.array([label], dtype=np.float32))


if __name__ == "__main__":
    omt = OmniglotTest(config.test_dir)

