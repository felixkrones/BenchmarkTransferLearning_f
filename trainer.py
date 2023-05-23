import numpy as np
from utils import MetricLogger, ProgressLogger
from models import ClassificationNet, build_classification_model
import time
import torch
from tqdm import tqdm
import warnings

from gmml.model_utils import metric_AUROC


def train_one_epoch(data_loader_train, device,model, criterion, optimizer, epoch):
  batch_time = MetricLogger('Time', ':6.3f')
  losses = MetricLogger('Loss', ':.4e')
  progress = ProgressLogger(
    len(data_loader_train),
    [batch_time, losses],
    prefix="Epoch: [{}]".format(epoch))

  model.train()

  end = time.time()
  for i, (samples, targets) in enumerate(data_loader_train):
    samples, targets = samples.float().to(device), targets.float().to(device)

    outputs = model(samples)
    
    #if torch.min(outputs) < 0:
    #  warnings.warn("Negative output detected. Sigmoid activation is applied.")
    #  outputs = torch.sigmoid(outputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.update(loss.item(), samples.size(0))
    batch_time.update(time.time() - end)
    end = time.time()

    if i % 50 == 0:
      progress.display(i)


def evaluate(data_loader_val, device, model, criterion, args):
  model.eval()

  with torch.no_grad():
    batch_time = MetricLogger('Time', ':6.3f')
    losses = MetricLogger('Loss', ':.4e')
    progress = ProgressLogger(
      len(data_loader_val),
      [batch_time, losses], prefix='Val: ')

    p_out = torch.FloatTensor().to(device)
    t_out = torch.FloatTensor().to(device)

    end = time.time()
    for i, (samples, targets) in enumerate(data_loader_val):
      samples, targets = samples.float().to(device), targets.float().to(device)

      outputs = model(samples)
      #if torch.min(outputs) < 0:
      #  outputs = torch.sigmoid(outputs)

      loss = criterion(outputs, targets)

      p_out = torch.cat((p_out, outputs), 0)
      t_out = torch.cat((t_out, targets), 0)

      losses.update(loss.item(), samples.size(0))
      losses.update(loss.item(), samples.size(0))
      batch_time.update(time.time() - end)
      end = time.time()

      if i % 50 == 0:
        progress.display(i)

    AUC_all = metric_AUROC(t_out, p_out)
    AUC_mean = np.mean(AUC_all)

    print(f"Validation AUC_mean: {AUC_mean:.4f}, AUC_all: {AUC_all}")
    if args is not None and args.data_set == "CheXpert":
      AUC_mean_5 = np.mean(np.array(AUC_all)[[2,5,6,8,10]])
      print(f"Validation AUC_mean_5: {AUC_mean_5:.4f}")

  return losses.avg


def test_classification(checkpoint, data_loader_test, device, args):
  if "vit" in args.model_name.lower():
    model = build_classification_model(args)
  else:
    model = ClassificationNet(args.model_name.lower(), args.num_class, args, activation=args.activate)
    
  #print(f'model to load weights in: {model}')
  
  modelCheckpoint = torch.load(checkpoint, map_location=device)
  if "state_dict" in modelCheckpoint:
      state_dict = modelCheckpoint["state_dict"]
  elif "model" in modelCheckpoint:
      state_dict = modelCheckpoint["model"]
  else:
      raise ValueError(f"No state_dict or model in modelCheckpoint: {modelCheckpoint.keys()}")
  
  for k in list(state_dict.keys()):
    if k.startswith('module.'):
      state_dict[k[len("module."):]] = state_dict[k]
      del state_dict[k]
    if k.startswith('vit.'):
      state_dict[k[len("vit."):]] = state_dict[k]
      del state_dict[k]
  for k in list(state_dict.keys()):
    if k.startswith('head_class.'):
      state_dict[f'head.{k[len("head_class."):]}'] = state_dict[k]
      del state_dict[k]

  if "vit" in args.model_name.lower():
    if state_dict['patch_embed.proj.weight'].shape[1] < model.state_dict()['patch_embed.proj.weight'].shape[1]:
        print(f"Number of channels in pretrained model {state_dict['patch_embed.proj.weight'].shape} is not same as the model {model.state_dict()['patch_embed.proj.weight'].shape}. Converting the pretrained model")
        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].repeat(1, model.state_dict()['patch_embed.proj.weight'].shape[1], 1, 1)
        print(f"New shape of pretrained model {state_dict['patch_embed.proj.weight'].shape}")

  print(f'state_dict to load: {state_dict.keys()}')

  msg = model.load_state_dict(state_dict, strict=False)
  print('Loaded with msg: {}'.format(msg))
  assert len(msg.missing_keys) == 0
  print("=> loaded pre-trained model '{}'".format(checkpoint))

  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  model.to(device)

  model.eval()

  y_test = torch.FloatTensor().to(device)
  p_test = torch.FloatTensor().to(device)

  with torch.no_grad():
    for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
      targets = targets.to(device)
      y_test = torch.cat((y_test, targets), 0)

      if len(samples.size()) == 4:
        bs, c, h, w = samples.size()
        n_crops = 1
      elif len(samples.size()) == 5:
        bs, n_crops, c, h, w = samples.size()

      varInput = torch.autograd.Variable(samples.view(-1, c, h, w).to(device))

      if "vit" in args.model_name.lower():
        out = model(varInput)
        if args.data_set == "RSNAPneumonia":
          out = torch.softmax(out,dim = 1)
        else:
          out = torch.sigmoid(out)
        outMean = out.view(bs, n_crops, -1).mean(1)
        p_test = torch.cat((p_test, outMean.data), 0)
      else:
        out = model(varInput)
        outMean = out.view(bs, n_crops, -1).mean(1)
        p_test = torch.cat((p_test, outMean.data), 0)
        
  return y_test, p_test

def test_segmentation(model, model_save_path,data_loader_test, device,log_writter):
    print("testing....", file=log_writter)
    checkpoint = torch.load(model_save_path)
    state_dict = checkpoint["state_dict"]
    for k in list(state_dict.keys()):
      if k.startswith('module.'):
        state_dict[k[len("module."):]] = state_dict[k]
        del state_dict[k]
    model.load_state_dict(state_dict)
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
    model.to(device)
    with torch.no_grad():
        test_p = None
        test_y = None
        model.eval()
        for batch_ndx, (x_t, y_t) in enumerate(tqdm(data_loader_test)):
            x_t, y_t = x_t.float().to(device), y_t.float().to(device)
            pred_t = model(x_t)
            if test_p is None and test_y is None:
                test_p = pred_t
                test_y = y_t
            else:
                test_p = torch.cat((test_p, pred_t), 0)
                test_y = torch.cat((test_y, y_t), 0)

            if (batch_ndx + 1) % 5 == 0:
                print("Testing Step[{}]: ".format(batch_ndx + 1) , file=log_writter)
                log_writter.flush()

        print("Done testing iteration!", file=log_writter)
        log_writter.flush()

    test_p = test_p.cpu().detach().numpy()
    test_y = test_y.cpu().detach().numpy()

    return test_y, test_p


