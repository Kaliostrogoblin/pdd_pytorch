import numpy as np
import torch
import time
import os

from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2
from torchvision import transforms
from torch import nn

from train_test_split import datadir_train_test_split
from data_utils import AllCropsDataset
from data_utils import SiameseSampler
from siamese import TwinNetwork
from siamese import TransferTwinNetwork
from siamese import SiameseNetwork

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_ZIP_PATH = 'archive_full.zip'
DATA_PATH = 'data/'
TEST_SIZE = 0.2
BATCH_SIZE = 32
STEPS_PER_EPOCH = 100
VALIDATION_STEPS = 30
EMBEDDING_DIM = 32
EPOCHS = 100
LR = 0.0001
RS = 13

# for reproducibility
np.random.seed(RS)
torch.manual_seed(RS)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def unzip_data():
    os.system("unzip %s -d %s" % (DATA_ZIP_PATH, DATA_PATH)) 


def split_on_train_and_test():
    for crop in os.listdir(DATA_PATH):
        crop_path = os.path.join(DATA_PATH, crop)
        _ = datadir_train_test_split(crop_path, 
                                    test_size=TEST_SIZE, 
                                    random_state=RS)


def prepare_datasets():
    train_ds = AllCropsDataset(
        DATA_PATH, 
        subset='train',
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        target_transform=torch.tensor)

    test_ds = AllCropsDataset(
        DATA_PATH, 
        subset='test',
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        target_transform=torch.tensor)

    # print statistics
    print('Train size:', len(train_ds))
    print('Test size:', len(test_ds))
    print('Number of samples in the dataset:', len(train_ds))
    print('Crops in the dataset:', train_ds.crops)
    print('Total number of classes in the dataset:', len(train_ds.classes))
    print('Classes with the corresponding targets:')
    print(train_ds.class_to_idx)
    return train_ds, test_ds


def train(model, 
          criterion, 
          optimizer, 
          train_data_loader, 
          test_data_loader):

    start = time.time()

    for i in range(EPOCHS):
        start_epoch = time.time()
        model.train()
        running_loss = 0
        running_acc = 0
        for b, batch in enumerate(train_data_loader):
            if b == STEPS_PER_EPOCH:
                break
            # unpack batch
            (batch_xs_l, batch_xs_r), batch_ys = batch
            batch_ys = batch_ys.float()

            # put into cpu or gpu
            batch_xs_l = batch_xs_l.to(DEVICE)
            batch_xs_r = batch_xs_r.to(DEVICE)
            batch_ys = batch_ys.to(DEVICE)

            # zero the parameter gradients
            optimizer.zero_grad()

            # calculate loss and do backward
            output = model(batch_xs_l, batch_xs_r).view(-1)
            loss = criterion(output, batch_ys)
            loss.backward()
            optimizer.step()
            
            # save loss for statistics
            running_loss += loss.item()
            # accuracy
            output = (output > 0.5).float()
            running_acc += (output == batch_ys).float().sum()
            
        # print train loss and train acc
        train_loss = running_loss / STEPS_PER_EPOCH
        train_acc = running_acc / (STEPS_PER_EPOCH*BATCH_SIZE)
        print("Epoch {}/{}, Loss: {:.4f}, Accuracy: {:.3f}".format(
            i+1, EPOCHS, train_loss, train_acc))
            
        # calculate test loss and accuracy
        model.eval()
        test_loss = 0
        test_acc = 0
        for b_test, batch in enumerate(test_data_loader):
            if b_test == VALIDATION_STEPS:
                break
            # unpack batch
            (batch_xs_l, batch_xs_r), batch_ys = batch
            batch_ys = batch_ys.float()

            # put into cpu or gpu
            batch_xs_l = batch_xs_l.to(DEVICE)
            batch_xs_r = batch_xs_r.to(DEVICE)
            batch_ys = batch_ys.to(DEVICE)
                
            with torch.no_grad():
                # calculate loss
                output = model(batch_xs_l, batch_xs_r).view(-1)
                test_loss += criterion(output, batch_ys).item()   

                # accuracy
                output = (output > 0.5).float()
                test_acc += (output == batch_ys).float().sum()
            
        # print train loss and train acc
        test_loss = test_loss / STEPS_PER_EPOCH
        test_acc = test_acc / (VALIDATION_STEPS*BATCH_SIZE)
        print("Epoch {}/{}:{:.2f}s, Test Loss: {:.4f}, Test Accuracy: {:.3f}".format(
            i+1, EPOCHS, time.time()-start_epoch, test_loss, test_acc))

    print('Took:{:.2f}'.format(time.time()-start))
    return model


def main():
    print("Extract data")
    unzip_data()

    print("Split on train and test")
    split_on_train_and_test()

    print("Create datasets")
    train_ds, test_ds = prepare_datasets()

    print("Create data loaders")
    train_sampler = SiameseSampler(train_ds, random_state=RS)
    test_sampler = SiameseSampler(test_ds, random_state=RS)
    train_data_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4)
    test_data_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, sampler=test_sampler, num_workers=4)

    print("Build computational graph")
    mobilenet = mobilenet_v2(pretrained=True)
    # remove last layer
    mobilenet = torch.nn.Sequential(*(list(mobilenet.children())[:-1]))
    siams = SiameseNetwork(
        twin_net=TransferTwinNetwork(
            base_model=mobilenet,
            output_dim=EMBEDDING_DIM))
    siams.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(siams.parameters(), lr=LR)

    print("Train model")
    siams = train(siams, criterion, optimizer, train_data_loader, test_data_loader)

    print("Save model")
    torch.save(siams.twin_net.state_dict(), 'models/twin.pt')


if __name__ == "__main__":
    main()
