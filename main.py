import torch
import os
import argparse
from torch.utils.data import DataLoader
import pickle
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import Model
from trainer import Trainer
import dataloader
import numpy as np

# GPU or CPU
if torch.cuda.is_available():  
  device = "cuda:0"
  print("running on gpu")
else:  
  device = "cpu"
  print("running on cpu")
  
''' Options '''

parser = argparse.ArgumentParser()
parser.add_argument("--label_dir", default = "/gpfsstore/rech/tvs/uki75tv/trab_patches_7param.csv", help = "path to label csv file")  #"./Train_Label_7p_lrhr.csv")
parser.add_argument("--image_dir", default = "/gpfsstore/rech/tvs/uki75tv/slice", help = "path to image directory")  #"./Train_LR_segmented")"
parser.add_argument("--mask_dir", default = "/gpfswork/rech/tvs/uki75tv/mask", help = "path to mask")
parser.add_argument("--tensorboard_name", default = "human", help = "give the name of your experiment for tensorboard")
parser.add_argument("--in_channel", type=int, default = 1, help = "nb of image channel")
parser.add_argument("--train_cross", default = "./cross_output.pkl", help = "filename of the output of the cross validation")
parser.add_argument("--batch_size", type=int, default = 8, help = "number of batch")
parser.add_argument("--model", default = "ConvNet", help="Choose model : Unet or ConvNet") 
parser.add_argument("--nof", type=int, default = 61, help = "number of filter")
parser.add_argument("--lr", type=float, default = 0.0001, help = "learning rate")
parser.add_argument("--nb_epochs", type=int, default = 150, help = "number of epochs")
parser.add_argument("--checkpoint_path", default = "./", help = "path to save or load checkpoint")
parser.add_argument("--mode", default = "train", help = "Mode used : Train, Using or Test")
parser.add_argument("--k_fold", type=int, default = 1, help = "Number of splitting for k cross-validation")
parser.add_argument("--n1", type=int, default = 141, help = "number of neurons in the first layer of the neural network")
parser.add_argument("--n2", type=int, default = 126, help = "number of neurons in the second layer of the neural network")
parser.add_argument("--n3", type=int, default = 80, help = "number of neurons in the third layer of the neural network")
parser.add_argument("--nb_workers", type=int, default = 0, help ="number of workers for datasets")
parser.add_argument("--NB_LABEL", type=int, default = 7, help = "specify the number of labels")
parser.add_argument("--optim", type=str, default = "Adam", help= "specify the optimizer")

opt = parser.parse_args()
NB_DATA = 34476
PERCENTAGE_TEST = 20
SIZE_IMAGE = 512
NB_LABEL = opt.NB_LABEL
'''functions'''

## Create summary for tensorboard
writer = SummaryWriter(log_dir='runs/'+opt.tensorboard_name)

## RESET WEIGHT FOR CROSS VALIDATION
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
    if hasattr(layer, 'reset_parameters'):
        print(f'Reset trainable parameters of layer = {layer}')
        layer.reset_parameters()

## FOR TRAINING

def train():
    #opt.n1 = trial.suggest_int('n1',90,500)
    #opt.n2 = trial.suggest_int('n2',100,500)
    #opt.n3 = trial.suggest_int('n3',100,500)
    #opt.lr = trial.suggest_loguniform('lr',1e-7,1e-3)
    #opt.nof = trial.suggest_int('nof',8,64)
    #opt.batch_size = trial.suggest_int('batch_size',8,24,step=8)
    
    # Create the folder where to save results and checkpoints
    save_folder=None
    i=0
    #while True:
    #    i += 1
        #if os.path.isdir("./result/TF_human_19µm"+str(i)) == False:
        #    save_folder = "./result/TF_human_19µm"+str(i)
        #    os.mkdir(save_folder)
        #    break
    #score_mse_t = []
    #score_mse_v = []
    #score_train_per_param = []
    #score_test_per_param = []
    # defining data
    index = range(NB_DATA)
    index_set = train_test_split(index,test_size=0.2,shuffle=False)
    #index_set=train_test_split(index,test_size=0.4,random_state=42)
    scaler = dataloader.normalization(opt.label_dir,index_set[0])
    #scaler = dataloader.normalization("/gpfswork/rech/tvs/uki75tv/BPNN/csv_files/Train_Label_7p_lrhr.csv",opt.norm_method,range(10500))
    #test_datasets = dataloader.Datasets(csv_file = "./Test_Label_6p.csv", image_dir="/gpfsstore/rech/tvs/uki75tv/Test_segmented_filtered", mask_dir = "/gpfsstore/rech/tvs/uki75tv/Test_trab_mask", scaler=scaler,opt=opt)
    
    #test_datasets = dataloader.Datasets(csv_file = "./Label_trab_FSRCNN.csv", image_dir="./TRAB_FSRCNN", mask_dir = "./MASK_FSRCNN", scaler=scaler,opt=opt, upsample=False)

    datasets = dataloader.Datasets(csv_file = opt.label_dir, image_dir = opt.image_dir, mask_dir = opt.mask_dir, scaler=scaler, opt=opt) # Create dataset
    print("start training")
    
    trainloader = DataLoader(datasets, batch_size = opt.batch_size, sampler = shuffle(index_set[0]), num_workers = opt.nb_workers )
    print(len(trainloader))
    testloader = DataLoader(datasets, batch_size = 1, num_workers = opt.nb_workers,sampler=index_set[1])#, shuffle=True)
    # defining the model
    if opt.model == "ConvNet":
        print("## Choose model : convnet ##")
        model = Model.ConvNet(in_channel=opt.in_channel,features =opt.nof,out_channels=NB_LABEL,n1=opt.n1,n2=opt.n2,n3=opt.n3,k1 = 3,k2 = 3,k3= 3).to(device)
    elif opt.model == "resnet50":
        print("## Choose model : resnet50 ##")
        model = Model.ResNet50(14,1).to(device)
    elif opt.model == "restnet101":
        print("## Choose model : resnet101 ##")
        model = Model.ResNet101(14,1).to(device)
    elif opt.model == "Unet":
        print("## Choose model : Unet ##")
        model = Model.UNet(in_channels=opt.in_channel,out_channels=1,nb_label=NB_LABEL, n1=opt.n1, n2=opt.n2, n3=opt.n3, init_features=opt.nof).to(device)
    elif opt.model == "MultiNet":
        print("## Choose model : MultiNet ##")
        model = Model.MultiNet(features =opt.nof,out_channels=NB_LABEL,n1=opt.n1,n2=opt.n2,n3=opt.n3,k1 = 3,k2 = 3,k3= 3).to(device)
    #torch.manual_seed(2)
    #model.apply(reset_weights)
    model.load_state_dict(torch.load("./convnet_pixel/BPNN_checkpoint_199.pth"))
    count = 0
    for name, param in model.named_parameters():
        if "conv" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
        count += 1
    #TF_model = Model.Add_TL(n1=80,n2=40,out_channels=7).to(device)
    # verify if freeze layer are correct
    print("Verify that freeze layer are:{}, and {}".format(False,3))
    for name, param in model.named_parameters():
        print(f'{name}: requires_grad={param.requires_grad}')
            
    # Start training
    t = Trainer(opt,model,device,save_folder,scaler=scaler)
    for epoch in range(opt.nb_epochs):
        mse_train, param_train = t.train(trainloader,epoch)
        mse_test, param_test = t.test(testloader,epoch,writer)
        writer.add_scalars('Loss',{'train':mse_train,'test':mse_test},epoch)
        writer.add_scalars('BioParam/euler_number',{'train':param_train[0],'test':param_test[0]},epoch)
        writer.add_scalars('BioParam/trabecular_thickness',{'train':param_train[1],'test':param_test[1]},epoch)
        writer.add_scalars('BioParam/trabecular_pattern_factor',{'train':param_train[2],'test':param_test[2]},epoch)
        writer.add_scalars('BioParam/bone_perimeteter_area_ratio',{'train':param_train[3],'test':param_test[3]},epoch)
        writer.add_scalars('BioParam/number_object',{'train':param_train[4],'test':param_test[4]},epoch)
        writer.add_scalars('BioParam/area',{'train':param_train[5],'test':param_test[5]},epoch)
        writer.add_scalars('BioParam/diameter',{'train':param_train[6],'test':param_test[6]},epoch)

        # score_mse_t.append(mse_train)
        # score_mse_v.append(mse_test)
        # score_train_per_param.append(param_train)
        # score_test_per_param.append(param_test)
    #resultat = {"mse_train":score_mse_t, "mse_test":score_mse_v,"train_per_param":score_train_per_param,"test_per_param":score_test_per_param}
    #with open(os.path.join(save_folder,opt.train_cross),'wb') as f:
    #    pickle.dump(resultat, f)
    writer.close()
    #with open(os.path.join(save_folder,"history.txt"),'wb') as g:
    #    history = "nof: " + str(opt.nof) + " model:" +str(opt.model) + " lr:" + str(opt.lr) + " neurons: " + str(opt.n1) + " " + str(opt.n2) + " " + str(opt.n3) + " kernel:" + str(3) + " norm data: " + str(opt.norm_method)
    #    pickle.dump(history,g)
      
''' main '''
if opt.mode == "train":
    train()
else :
    i=0
    while True:
        i += 1
        if os.path.isdir("./result/test_FSRCNN_"+str(i)) == False:
            save_folder = "./result/test_FSRCNN_"+str(i)
            os.mkdir(save_folder)
            break
    
    # model #
    index = list(range(NB_DATA))
    scaler = dataloader.normalization(opt.label_dir,opt.norm_method,index)
    datasets = dataloader.Datasets(csv_file = "./Label_trab_FSRCNN.csv", image_dir="./TRAB_FSRCNN", mask_dir = "./MASK_FSRCNN", scaler=scaler,opt=opt, upsample=False)
    #datasets = dataloader.Datasets(csv_file = "./Test_Label_6p.csv", image_dir="/gpfsstore/rech/tvs/uki75tv/Test_segmented_filtered", mask_dir = "/gpfsstore/rech/tvs/uki75tv/Test_trab_mask", scaler=scaler,opt=opt, upsample=False)
    #index_human = range(400)
    #index_set=train_test_split(index_human,test_size=0.90,random_state=42)
    model = Model.ConvNet(in_channel=opt.in_channel,features =opt.nof,out_channels=NB_LABEL,n1=opt.n1,n2=opt.n2,n3=opt.n3,k1 = 3,k2 = 3,k3= 3).to(device)
    #scaler = dataloader.normalization("./Train_Label_6p_augment.csv", opt.norm_method,index)
    testloader = DataLoader(datasets, batch_size = 1, num_workers = opt.nb_workers)
    t = Trainer(opt,model,device,save_folder,scaler)
    t.test(testloader,opt.nb_epochs)
  
