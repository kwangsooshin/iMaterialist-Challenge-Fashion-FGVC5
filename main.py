import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from FGVC5_Fashion import FashionDataset
from torch.utils.data import DataLoader




class Models:
    class __logging__:
        def __init__(self):
            import logging
            self.__log = logging.getLogger("")
            __log_format = logging.Formatter('[%(levelname)s] %(asctime)s > %(message)s')
            __log_handler = logging.FileHandler('./logfile')
            __log_handler.setFormatter(__log_format)
            self.__log.addHandler(__log_handler)


        def critical(self, message):
            print(message)
            self.__log.critical(message)

        def error(self, message):
            print(message)
            self.__log.error(message)

        def warning(self, message):
            print(message)
            self.__log.warning(message)

        def info(self, message):
            print(message)
            self.__log.info(message)

        def debug(self, message):
            print(message)
            self.__log.debug(message)


    def __init__(self):
        self.log = self.__logging__()
        self.log.info("Init Model")

        model = models.densenet161(pretrained=False)

        classifier = nn.Sequential()
        classifier.add_module("Linear" ,nn.Linear(model.classifier.in_features, 228))
        # classifier.add_module("Softmax", nn.Softmax())
        model.classifier = classifier

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.model = nn.DataParallel(model)
            self.model = self.model.cuda()

        self.dataloader()

        self.loss_fun = F.binary_cross_entropy_with_logits
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                   lr=0.01, weight_decay=1e-4, momentum=0.9, nesterov=True)


    def train(self):
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            data, target = Variable(data), Variable(target)
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            self.optimizer.zero_grad()
            output = self.model.forward(data)
            loss = self.loss_fun(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.data.mean()
            self.log.info("\t- Batch: {}/{}, Loss: {}".format(int(batch_idx), len(self.train_dataloader), loss.data.item()))
        self.log.info("\t> Average Loss: {}".format(total_loss / len(self.train_dataloader)))


    def test(self):
        total_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.val_dataloader):
            data, target = Variable(data), Variable(target)
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = self.model.forward(data)
            loss = self.loss_fun(output, target)
            total_loss += loss.data.item()
        self.log.info("\t> Error: {0}".format(100.0 * total_loss / len(self.val_dataloader.dataset)))


    def dataloader(self):
        train_data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        val_data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        train_dataset = FashionDataset(csv_file='/workspace/dataset/FGVC5_Fashion/train.csv',
                                       image_dir='/workspace/dataset/FGVC5_Fashion/train',
                                       transform=train_data_transforms)

        val_dataset = FashionDataset(csv_file='/workspace/dataset/FGVC5_Fashion/validation.csv',
                                     image_dir='/workspace/dataset/FGVC5_Fashion/validation',
                                     transform=val_data_transforms)

        self.train_dataloader = DataLoader(train_dataset, batch_size=80, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=40, shuffle=True)

        self.log.info("Data Loader Complete")



if __name__ == "__main__":
    test = Models()
    for i in range(30):
        test.train()
    test.test()
