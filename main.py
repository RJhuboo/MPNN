import torch
import os
from matplotlib import pyplot as plt
import argparse
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import Model
from trainer import Trainer
import dataloader
from torch.utils.tensorboard import SummaryWriter


# GPU or CPU
if torch.cuda.is_available():  
  device = "cuda:0"
  print("running on gpu")
else:  
  device = "cpu"
  print("running on cpu")
  
''' Options '''

parser = argparse.ArgumentParser()
parser.add_argument("--label_dir", default = "/gpfsstore/rech/tvs/uki75tv/Trab2D_lrhr_7p.csv", help = "path to label csv file")
parser.add_argument("--image_dir", default = "/gpfsstore/rech/tvs/uki75tv/Train_LR_segmented")#"/gpfsstore/rech/tvs/uki75tv/DATA_HUMAN/IMAGE/", help = "path to image directory")
parser.add_argument("--mask_dir", default = "/gpfsstore/rech/tvs/uki75tv/mask", help = "path to mask")
parser.add_argument("--in_channel", type=int, default = 1, help = "nb of image channel")
parser.add_argument("--train_cross", default = "./cross_output.pkl", help = "filename of the output of the cross validation")
parser.add_argument("--batch_size", type=int, default = 24, help = "number of batch")
parser.add_argument("--nof", type=int, default = 64, help = "number of filter")
parser.add_argument("--lr", type=float, default = 0.0002, help = "learning rate")
parser.add_argument("--nb_epochs", type=int, default = 300, help = "number of epochs")
parser.add_argument("--checkpoint_path", default = "./result", help = "path to save or load checkpoint")
parser.add_argument("--mode", default = "train", help = "Mode used : Train, Using or Test")
parser.add_argument("--n1", type=int, default = 158, help = "number of neurons in the first layer of the neural network")
parser.add_argument("--n2", type=int, default = 152, help = "number of neurons in the second layer of the neural network")
parser.add_argument("--n3", type=int, default = 83, help = "number of neurons in the third layer of the neural network")
parser.add_argument("--nb_workers", type=int, default = 0, help ="number of workers for datasets")
parser.add_argument("--norm_method", type=str, default = "standardization", help = "Choose how to normalize bio parameters")
parser.add_argument("--NB_LABEL", type=int, default = 7, help = "Specify the number of labels")
parser.add_argument("--optim", type=str, default = "Adam", help= "Specify the optimizer")
parser.add_argument("--tensorboard_name", default = "Pixel", help = "Name to your experiment on tensorboard")

opt = parser.parse_args()

NB_DATA = 14700
PERCENTAGE_TEST = 20
SIZE_IMAGE = 512

# Open writer of tensorboard using pytorch
writer = SummaryWriter(log_dir='runs/'+opt.tensorboard_name)

''' TRAINING '''

def train():
  
    # Create the folder where to save results and checkpoints
    save_folder = opt.checkpoint_path
    if os.path.isdir(save_folder) == False:
        os.mkdir(save_folder)
      
    score_mse_t = []
    score_mse_v = []
    score_train_per_param = []
    score_test_per_param = []

    # Split data into 2 parts
    index = range(NB_DATA)
    split = train_test_split(index,test_size = 0.2,shuffle=False)
    train_index = split[0]
    test_index = split[1]

    # Create Train and Test loader
    scaler = dataloader.normalization(opt.label_dir,opt.norm_method,train_index)
    datasets = dataloader.Datasets(csv_file = opt.label_dir, image_dir = opt.image_dir, mask_dir = opt.mask_dir, scaler=scaler, opt=opt) # Create dataset
    trainloader = DataLoader(datasets, batch_size = opt.batch_size, sampler = shuffle(train_index), num_workers = opt.nb_workers )
    testloader = DataLoader(datasets,batch_size = 1, sampler = test_index,num_workers = opt.nb_workers)#test_datasets, batch_size = 1, num_workers = opt.nb_workers, shuffle=True)
    
    # Defining the model
    print("## Load MPNN ##")
    model = Model.ConvNet(in_channel=opt.in_channel,features =opt.nof,out_channels=opt.NB_LABEL,n1=opt.n1,n2=opt.n2,n3=opt.n3,k1 = 3,k2 = 3,k3= 3).to(device)
    torch.manual_seed(2)

    # Start training
    t = Trainer(opt,model,device,save_folder,scaler)
    for epoch in range(opt.nb_epochs):
        mse_train, param_train = t.train(trainloader,epoch)
        mse_test, param_test = t.test(testloader,writer,epoch)
        writer.add_scalars('Loss',{'train':mse_train, 'test':mse_test},epoch)    
        writer.add_scalars('Loss',{'train':mse_train,'test':mse_test},epoch)
        writer.add_scalars('BioParam/euler_number',{'train':param_train[0],'test':param_test[0]},epoch)
        writer.add_scalars('BioParam/trabecular_thickness',{'train':param_train[1],'test':param_test[1]},epoch)
        writer.add_scalars('BioParam/trabecular_pattern_factor',{'train':param_train[2],'test':param_test[2]},epoch)
        writer.add_scalars('BioParam/bone_perimeteter_area_ratio',{'train':param_train[3],'test':param_test[3]},epoch)
        writer.add_scalars('BioParam/number_object',{'train':param_train[4],'test':param_test[4]},epoch)
        writer.add_scalars('BioParam/area',{'train':param_train[5],'test':param_test[5]},epoch)
        writer.add_scalars('BioParam/diameter',{'train':param_train[6],'test':param_test[6]},epoch)
    writer.close()

def test():
    ''' Must be modified in function of the inference set '''
    if os.path.isdir("./result/test") == False:
        save_folder = "./result/test"
        os.mkdir(save_folder)
      
    # Model
    index = list(range(NB_DATA))
    scaler = dataloader.normalization(opt.label_dir,opt.norm_method,index)
  
    datasets = dataloader.Datasets(csv_file = "../Trab_Human.csv", image_dir="../DATA_HUMAN/IMAGE/", mask_dir = "../DATA_HUMAN/MASK/", scaler=scaler,opt=opt, upsample=False)
    index_human = range(400)
    index_set=train_test_split(index_human,test_size=0.90,random_state=42)
    model = Model.ConvNet(in_channel=opt.in_channel,features =opt.nof,out_channels=opt.NB_LABEL,n1=opt.n1,n2=opt.n2,n3=opt.n3,k1 = 3,k2 = 3,k3= 3).to(device)
    testloader = DataLoader(datasets, batch_size = 1, num_workers = opt.nb_workers,sampler=index_set[1])
    t = Trainer(opt,model,device,save_folder,scaler)
    t.test(testloader,opt.nb_epochs)
  
''' main '''
if opt.mode == "train":
    train()
else:
    test()
    
