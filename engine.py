
import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from tqdm import tqdm
import copy
from models import build_classification_model, save_checkpoint
from utils import metric_AUROC
from sklearn.metrics import accuracy_score
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from trainer import train_one_epoch,test_classification,evaluate,test_segmentation
import segmentation_models_pytorch as smp
from utils import cosine_anneal_schedule,dice,mean_dice_coef

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler


sys.setrecursionlimit(40000)


def classification_engine(args, model_path, output_path, diseases, dataset_train, dataset_val, dataset_test, test_diseases=None):
  device = torch.device(args.device)
  cudnn.benchmark = True

  model_path = os.path.join(model_path, args.exp_name)

  if not os.path.exists(model_path):
    os.makedirs(model_path)

  if not os.path.exists(output_path):
    os.makedirs(output_path)


  # training phase
  if args.mode == "train":
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.workers, pin_memory=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True)
    log_file = os.path.join(model_path, "models.log")

    # training phase
    print("start training....")
    for i in range(args.start_index, args.num_trial):
      print ("run:",str(i+1))
      start_epoch = 0
      init_loss = 1000000
      experiment = args.exp_name + "_run_" + str(i)
      best_val_loss = init_loss
      patience_counter = 0
      save_model_path = os.path.join(model_path, experiment)
      if ("vit" in args.model_name.lower()) or ("swin" in args.model_name.lower()):
        criterion = torch.nn.BCEWithLogitsLoss()
        if args.data_set == "RSNAPneumonia":
          criterion = torch.nn.CrossEntropyLoss()
      else:
        criterion = torch.nn.BCELoss()

      model = build_classification_model(args)

      if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
      model.to(device)

      parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

      if ("vit" in args.model_name.lower()) or ("swin" in args.model_name.lower()):
        optimizer = create_optimizer(args, model)
        loss_scaler = NativeScaler()
        lr_scheduler, _ = create_scheduler(args, optimizer)
      else:
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=args.patience // 2, mode='min',
                                       threshold=0.0001, min_lr=0, verbose=True)


      if args.resume:
        resume = os.path.join(model_path, experiment + f'_{args.best}.pth.tar')
        if os.path.isfile(resume):
          print("=> loading checkpoint '{}'".format(resume))
          checkpoint = torch.load(resume)

          start_epoch = checkpoint['epoch']
          init_loss = checkpoint['lossMIN']
          model.load_state_dict(checkpoint['state_dict'])
          lr_scheduler.load_state_dict(checkpoint['scheduler'])
          optimizer.load_state_dict(checkpoint['optimizer'])
          print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={:.5f})"
                .format(resume, start_epoch, init_loss))
        else:
          print("=> no checkpoint found at '{}'".format(resume))



      for epoch in range(start_epoch, args.num_epoch):
        print(f"Training epoch: [{epoch}/{args.num_epoch}]")
        start_time_epoch = time.time()
        train_one_epoch(data_loader_train,device, model, criterion, optimizer, epoch)
        print(f"Trained epoch: [{epoch}/{args.num_epoch}] for {round((time.time()-start_time_epoch)/60, 3)} minutes")
        val_loss = evaluate(data_loader_val, device,model, criterion, args)

        lr_scheduler.step(val_loss)

        if val_loss < best_val_loss:
          save_checkpoint({
            'epoch': epoch + 1,
            'lossMIN': val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
          },  filename=save_model_path+"_best")
          save_checkpoint({
            'epoch': epoch + 1,
            'lossMIN': val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': lr_scheduler.state_dict(),
          },  filename=save_model_path+"_last")

          best_val_loss = val_loss
          patience_counter = 0

          print(
            "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_val_loss, val_loss,
                                                                                               save_model_path))

        else:
          print("Epoch {:04d}: val_loss did not improve from {:.5f} ".format(epoch, best_val_loss ))
          patience_counter += 1

        if patience_counter > args.patience:
          print("Early Stopping")
          break


      # log experiment
      with open(log_file, 'a') as f:
        f.write(experiment + "\n")
        f.close()

  print ("start testing.....")
  output_file = os.path.join(output_path, args.exp_name + "_results.txt")

  data_loader_test = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)

  log_file = os.path.join(model_path, "models.log")
  load_pretrained = False
  if not os.path.isfile(log_file):
    if args.proxy_dir is None:
      raise FileNotFoundError("log_file ({}) not exists!".format(log_file))
    else:
      print("Using procx_dir as log_file")
      saved_model = args.proxy_dir
      load_pretrained = True
      with open(log_file, 'a') as f:
        experiment = args.proxy_dir.replace(".pth.tar", "")
        f.write(experiment + "\n")
        f.close()
  mean_auc_test = []
  mean_auc_all = []
  accuracy = []
  with open(log_file, 'r') as reader, open(output_file, 'a') as writer:
    experiment = reader.readline()
    print(">> Disease = {}".format(diseases))
    writer.write("Disease = {}\n".format(diseases))

    while experiment:
      experiment = experiment.replace('\n', '')
      if load_pretrained:
        saved_model = os.path.join(args.proxy_dir)
      else:
          if model_path in experiment:
            saved_model = os.path.join(experiment + f"_{args.best}.pth.tar")
          else:
            saved_model = os.path.join(model_path, experiment + f"_{args.best}.pth.tar") 

      y_test, p_test = test_classification(saved_model, data_loader_test, device, args)
      all_results = metric_AUROC(y_test, p_test, args.num_class)

      if args.data_set == "RSNAPneumonia":
        acc = accuracy_score(np.argmax(y_test.cpu().numpy(),axis=1),np.argmax(p_test.cpu().numpy(),axis=1))
        print(">>{}: ACCURACY = {}".format(experiment,acc))
        writer.write(
          "{}: ACCURACY = {}\n".format(experiment, np.array2string(np.array(acc), precision=4, separator='\t')))
        accuracy.append(acc)
      if test_diseases is not None:
        y_test = copy.deepcopy(y_test[:,test_diseases])
        p_test = copy.deepcopy(p_test[:, test_diseases])
        individual_results = metric_AUROC(y_test, p_test, len(test_diseases))
      else:
        individual_results = metric_AUROC(y_test, p_test, args.num_class)
      print(">>{}: AUC test = {}".format(experiment, np.array2string(np.array(individual_results), precision=4, separator=',')))
      writer.write(
        "{}: AUC test = {}\n".format(experiment, np.array2string(np.array(individual_results), precision=4, separator='\t')))
      print(">>{}: AUC all = {}".format(experiment, np.array2string(np.array(all_results), precision=4, separator=',')))
      writer.write(
        "{}: AUC all = {}\n".format(experiment, np.array2string(np.array(all_results), precision=4, separator='\t')))


      mean_over_test_classes = np.array(individual_results).mean()
      mean_over_all_classes = np.array(all_results).mean()
      print(">>{}: AUC test = {:.4f}".format(experiment, mean_over_test_classes))
      writer.write("{}: AUC test = {:.4f}\n".format(experiment, mean_over_test_classes))
      print(">>{}: AUC all = {:.4f}".format(experiment, mean_over_all_classes))
      writer.write("{}: AUC all = {:.4f}\n".format(experiment, mean_over_all_classes))

      mean_auc_test.append(mean_over_test_classes)
      mean_auc_all.append(mean_over_all_classes)
      experiment = reader.readline()

    mean_auc_test = np.array(mean_auc_test)
    mean_auc_all = np.array(mean_auc_all)
    print(">> All trials: test mAUC  = {}".format(np.array2string(mean_auc_test, precision=4, separator=',')))
    writer.write("All trials: test mAUC  = {}\n".format(np.array2string(mean_auc_test, precision=4, separator='\t')))
    print(">> Mean AUC over All test trials: = {:.4f}".format(np.mean(mean_auc_test)))
    writer.write("Mean AUC over All test trials = {:.4f}\n".format(np.mean(mean_auc_test)))
    print(">> STD over All test trials:  = {:.4f}".format(np.std(mean_auc_test)))
    writer.write("STD over All test trials:  = {:.4f}\n".format(np.std(mean_auc_test)))

    print(">> All trials: all mAUC  = {}".format(np.array2string(mean_auc_all, precision=4, separator=',')))
    writer.write("All trials: all mAUC  = {}\n".format(np.array2string(mean_auc_all, precision=4, separator='\t')))
    print(">> Mean AUC over All all trials: = {:.4f}".format(np.mean(mean_auc_all)))
    writer.write("Mean AUC over All all trials = {:.4f}\n".format(np.mean(mean_auc_all)))
    print(">> STD over All all trials:  = {:.4f}".format(np.std(mean_auc_all)))
    writer.write("STD over All all trials:  = {:.4f}\n".format(np.std(mean_auc_all)))

def segmentation_engine(args, model_path, dataset_train, dataset_val, dataset_test,criterion):
  device = torch.device(args.device)
  if not os.path.exists(model_path):
    os.makedirs(model_path)

  logs_path = os.path.join(model_path, "Logs")
  if not os.path.exists(logs_path):
    os.makedirs(logs_path)

  if os.path.exists(os.path.join(logs_path, "log.txt")):
    log_writter = open(os.path.join(logs_path, "log.txt"), 'a')
  else:
    log_writter = open(os.path.join(logs_path, "log.txt"), 'w')

  if args.mode == "train":
    start_num_epochs=0
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                                 num_workers=args.train_num_workers)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.train_batch_size,
                                                      shuffle=False, num_workers=args.train_num_workers)
    if args.init.lower() == "imagenet":
      model = smp.Unet(args.backbone, encoder_weights=args.init, activation=args.activate)
    else:
      model = smp.Unet(args.backbone, encoder_weights=args.proxy_dir, activation=args.activate)

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
    model.to(device)
    best_val_loss = 100000
    patience_counter = 0

    for epoch in range(start_num_epochs, args.epochs):
        train_one_epoch(data_loader_train,device, model, criterion, optimizer, epoch)
        val_loss = evaluate(data_loader_val, device,model, criterion, args)
        # update learning rate
        lr_ = cosine_anneal_schedule(epoch,args.epochs,args.learning_rate)
        for param_group in optimizer.param_groups:
          param_group['lr'] = lr_
        if val_loss < best_val_loss:
          torch.save({
            'epoch': epoch + 1,
            'lossMIN': best_val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
          },  os.path.join(model_path, "checkpoint.pt"))

          best_val_loss = val_loss
          patience_counter = 0

          print(
            "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epoch, best_val_loss, val_loss,                                                                                      os.path.join(model_path,"checkpoint.pt")))
        else:
          print("Epoch {:04d}: val_loss did not improve from {:.5f} ".format(epoch, best_val_loss ))
          patience_counter += 1

        if patience_counter > args.patience:
          print("Early Stopping")
          break

        log_writter.flush()
  eval(f"torch.{args.device}.empty_cache()")

  data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False,
                                                num_workers=args.test_num_workers)
  test_y, test_p = test_segmentation(model, os.path.join(model_path, "checkpoint.pt"), data_loader_test, device, log_writter)

  print("[INFO] Dice = {:.2f}%".format(100.0 * dice(test_p, test_y)), file=log_writter)
  print("Mean Dice = {:.4f}".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter)
  log_writter.flush()


