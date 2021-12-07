import config
from dataset import OmniglotTrain, OmniglotTest
from torch.utils.data import DataLoader 
import torch 
from torch import nn 
from tqdm import tqdm 
from models.resnet_18 import SiameseResentNetwork
from models.resnet_cbam import SiameseResentCBAM
from models.model import initialize_weights
import numpy as np 
from  torch.optim.lr_scheduler import ExponentialLR
from network import SiameseUNet

class Train():
    def __init__(self):
        print("Loading dataset")
        self.train_loader = DataLoader(
            OmniglotTrain(config.train_dir),
            batch_size=config.batch_size, 
            shuffle=False,
            num_workers= 6 
        )

        self.test_loader = DataLoader(
            OmniglotTest(config.test_dir),
            batch_size=config.way, 
            shuffle=False,
            num_workers= 6 
        )

        self.model = SiameseUNet(img_dim=1, unet_type="attention").to(config.device)
        #self.model.apply(initialize_weights)

        if config.multi_gpus:
            self.model = torch.nn.DataParallel(self.model)
        
        if config.load_model:
            pass 

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.L2_Norm_loss = torch.nn.MSELoss()
        self.adam = torch.optim.Adam(self.model.parameters(), 
                                    lr = config.learning_rate, 
                                    weight_decay=5e-5)

        self.scheduler = ExponentialLR(self.adam, gamma=0.9)

    def test(self):
        correct = 0
        error = 0

        self.model.eval()
        with torch.no_grad():
            for img1, img2, label in self.test_loader:
                img1, img2, label = (img1.to(config.device), 
                                    img2.to(config.device), label.to(config.device))

                out = self.model(img1, img2)
                out = out.cpu().detach().numpy()
                pred = np.argmax(out)
                
                if pred == 0: correct += 1
                else: error += 1

        self.model.train()
        print("Total correct {} | wrong: {} | precision: {:0.4f}| accuracy: {:0.4f}".
                format(correct, error, correct*1.0/(correct + error), correct*1.0/config.times))
 


    def train(self):
        train_loop = tqdm(self.train_loader, leave=False)
        for epoch in range(config.num_epochs):
            train_loss = 0
            train_l2_loss = 0

            for img1, img2, label in train_loop:
                img1 = img1.to(config.device)
                img2 = img2.to(config.device)
                label = label.to(config.device)

                self.adam.zero_grad()
                re1, re2, out = self.model(img1, img2)

                id_loss = self.criterion(out, label)
                l2_loss = self.L2_Norm_loss(re1, img1) + self.L2_Norm_loss(re2, img2)
                loss = id_loss + l2_loss

                train_l2_loss += l2_loss.item()
                train_loss += loss.item()
                loss.backward()
                self.adam.step()

            print("Epoch ID: {} | train loss: {:.4f} | l2_loss {:.4f}".format(
                    epoch, train_loss / len(self.train_loader), train_l2_loss / len(self.train_loader)))
            self.test()

            if (epoch !=0 and epoch % 50 == 0):
                self.scheduler.step()
                print("learning rate ", self.adam.param_groups[0]["lr"])
                

if __name__ == "__main__":
    t = Train()
    t.train()