from comet_ml import Experiment
from sklearn.model_selection import train_test_split
from age_gender import Age_Gender_Model
from imdb import IMDBDataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import RandomHorizontalFlip, ToTensor, RandomCrop, Resize, CenterCrop
import argparse
from config import cfg, cfg_from_file

def init_model(cfg, device):
  print("Creating Model...")
  model = Age_Gender_Model()
  model.to(device)
  return model
def init_loaders(x, y, cfg):
  print("Loading Datasets...")
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=(1-cfg.TRAIN.TRAIN_RATIO), random_state=42)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

  train_data = IMDBDataset(X_train, y_train, root_dir=cfg.DATASET.DATA_FOLDER, transform=transforms.Compose([
                                               Resize(224),
                                               RandomHorizontalFlip(0.5),
                                               RandomCrop(224),
                                               ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                                           ]))
  val_data = IMDBDataset(X_val, y_val, root_dir=cfg.DATASET.DATA_FOLDER, transform=transforms.Compose([
                                               Resize(224),
                                               CenterCrop(224),
                                               ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                                           ]))
  test_data = IMDBDataset(X_test, y_test, root_dir=cfg.DATASET.DATA_FOLDER, transform=transforms.Compose([
                                               Resize(224),
                                               CenterCrop(224),
                                               ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                                           ]))
  dataloaders = dict()
  print("Loading dataloaders...")
  dataloaders['train'] = DataLoader(train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
  dataloaders['val'] = DataLoader(val_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False)
  dataloaders['test']= DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False)

  return dataloaders

def init_optimizers(cfg, model):
  
  print("Loading optimization variables")
  optimizer = torch.optim.Adam(model.parameters(),
                             lr=cfg.TRAIN.LEARNING_RATE,
                             betas=(0.9, 0.999),
                             eps=1e-08,
                             weight_decay=0,
                             amsgrad=False)
  
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience = cfg.TRAIN.PATIENCE )
  criterion = nn.CrossEntropyLoss()

  return {'criterion': criterion, 'optimizer': optimizer, 'scheduler': scheduler}

def training_loop(x,y, cfg, exp_name="exp_1", device=torch.device('cuda'),starting_epoch = 1):
  model = init_model(cfg, device)
  dataloaders = init_loaders(x,y,cfg)
  optim = init_optimizers(cfg, model)
  n_epochs = cfg.TRAIN.MAX_EPOCHS
  experiment = Experiment(project_name='mbm-pos-metrics',api_key = 'w4JbvdIWlas52xdwict9MwmyH')
  experiment.set_name(exp_name)
  hyper = {'train_batch_size': cfg.TRAIN.BATCH_SIZE,
           'age_lambda': cfg.MODEL.AGE_LAMBDA,
           'n_epochs': cfg.TRAIN.MAX_EPOCHS,
           'learning_rate': cfg.TRAIN.LEARNING_RATE}

  experiment.log_parameters(hyper)
  print("Starting training loop!")
  for epoch in range(starting_epoch, n_epochs + 1):
    for phase in ['train', 'val']:
      print("Epoch {}/{} - {}".format(epoch, n_epochs, phase.upper()))
      forward_model(model, dataloaders[phase], optim, cfg, device, experiment, phase, current_epoch = epoch)
    # checkpoint model
    torch.save(model,"checkpoints/{}_{}.pth".format(exp_name, epoch))
  experiment.end()
  return experiment
import time

def forward_model(model, data_loader, optim, cfg, device,
                  experiment,
                  phase = 'train',
                  scheduler = None,
                  current_epoch = 1):
  # n_epochs
  # criterion
  # scheduler
  # optimizer
  # device
  if phase == 'train':
    model.train()
  else:
    model.eval()
  
  n_batches = len(data_loader)
  running_acc_gender = 0.0
  running_acc_age = 0.0
  running_count = 0.0
  running_loss = 0.0
  running_time = 0.0
  start_time = time.time()

  for i, (img, target_gender, target_age) in enumerate(data_loader):

    #forward_batch
    optim['optimizer'].zero_grad()
    img = img.to(device)
    target_gender = target_gender.to(device)
    target_age = target_age.to(device)
    bs, *_ = img.shape 
    gen_pred, age_pred = model(img)
    loss = (1-cfg.MODEL.AGE_LAMBDA)*optim['criterion'](gen_pred, target_gender) + cfg.MODEL.AGE_LAMBDA*optim['criterion'](age_pred, target_age)
    loss.backward()
    optim['optimizer'].step()
    #print_results()
    running_loss += loss.item() * bs
    _, gen_preds = torch.max(gen_pred, 1)
    _, age_preds = torch.max(age_pred, 1)
    running_acc_gender += torch.sum(gen_preds == target_gender.data)
    running_acc_age += torch.sum(age_preds == target_age.data)
    running_count += bs
    running_time = time.time() - start_time
    projected_time = running_time/(i+1)*n_batches
    running_loss_m = running_loss/running_count
    accuracy_gender_m = 100*float(running_acc_gender)/float(running_count)
    accuracy_age_m = 100*float(running_acc_age)/float(running_count)
    print("\rBatch {}/{} --- Avg Loss: {:.4f} --- Gender Acc: {:.2f}% --- Age Acc: {:.2f}% --- Elapsed time: {}m{}s --- Projected time: {}m{}s ".format(i+1,
                                                                               n_batches,
                                                                               running_loss_m,
                                                                               accuracy_gender_m,
                                                                               accuracy_age_m,
                                                                               int(running_time//60), int(running_time)%60,
                                                                               int(projected_time//60), int(projected_time)%60
                                                                               ),end="")
    metrics = {"running_loss": running_loss_m,
               "accuracy_gender": accuracy_gender_m,
               "accuracy_age": accuracy_age_m,
               "epoch": current_epoch}

  print("")
  if phase == 'train':
    with experiment.train():
      experiment.log_metrics(metrics, epoch = current_epoch)
  elif phase == 'val':
    with experiment.validate():
      experiment.log_metrics(metrics, epoch = current_epoch)
  else:
    with experiment.test():
      experiment.log_metrics(metrics, epoch = current_epoch)
    #log_results()

  if phase == 'val':
    optim['scheduler'].step(running_loss_m)

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', dest='cfg_file', help='optional config file', type=str)
  parser.add_argument('--gpu',  dest='gpu_id', type=str, default='0')
  parser.add_argument('--name', type=str, default='experiment1')
  parser.add_argument('--features_path', dest='features_path', type=str, default='')
  parser.add_argument('--manualSeed', type=int, help='manual seed')
  args = parser.parse_args()
  return args

args = parse_args()

if args.cfg_file is not None:
  cfg_from_file(args.cfg_file)

data = torch.load(cfg.PREPROCESS.FEATURES)
e = training_loop(data['img'], data['target'], cfg, exp_name = args.name)
