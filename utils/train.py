import torch
import torchvision
import time
import psutil
import os
import numpy as np
from monai.losses import DiceLoss
import datetime
import pytz

def load_metrices(path):
  metrices_dir = path
  #if n : metrices_dir = OUT_DIR + 'tries/try' + str(n) + "/"
  train_loss = np.load(os.path.join(metrices_dir, 'loss_train.npy'))
  train_metric = np.load(os.path.join(metrices_dir, 'metric_train.npy'))
  test_loss = np.load(os.path.join(metrices_dir, 'loss_test.npy'))
  test_metric = np.load(os.path.join(metrices_dir, 'metric_test.npy'))
  return train_loss, train_metric, test_loss, test_metric

def dice_metric(predicted, target):
    '''
    In this function we take `predicted` and `target` (label) to calculate the dice coeficient then we use it 
    to calculate a metric value for the training and the validation.
    '''
    dice_value = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
    value = 1 - dice_value(predicted, target).item()
    return value


def get_time():
  utc_time= datetime.datetime.now(pytz.utc)
  local_time = utc_time.astimezone(pytz.timezone('Asia/Colombo'))
  return local_time.strftime("%Y:%m:%d %H:%M:%S")

def update_history(data,model_dir):
  history_file_path = model_dir + "history.csv"
  if not os.path.exists(history_file_path):
    with open(history_file_path,'a') as fd:
        fd.write(",".join(["Start", "End", "Best Matrix", "Best M. At", "Time Taken", "CUDA Memory Used", "CPU Memory","Time"]))
  with open(history_file_path,'a') as fd:
      str_data=[str(x) for x in (data + [get_time()])]
      fd.write("\n" + ",".join(str_data))

def train(model, data_in, loss, optim, max_epochs, model_dir, test_interval=1 , device=torch.device("cuda:0"), start_from=1):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []
    if (start_from != 1):
      save_loss_train, save_metric_train, save_loss_test, save_metric_test= [x.tolist() for x in load_metrices(model_dir)]
      best_metric = max(save_metric_train)
      best_metric_epoch = -2
    train_loader, test_loader = data_in

   
    start = time.time()
    max_gpu_memory = 0
    max_cpu_memory = 0
    gpu_memory_start = torch.cuda.max_memory_allocated(device=device)
    cpu_memory_start = psutil.Process().memory_info().rss

    for epoch in range(start_from -1 , max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0
        for batch_data in train_loader:
            
            train_step += 1

            volume = batch_data["image"]
            label = batch_data["label"]
            label = label != 0

            if torch.cuda.is_available():
              volume, label = (volume.to(device), label.to(device))

            optim.zero_grad()
        
            outputs = model(volume)
            
            train_loss = loss(outputs, label)
            
            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()
            print(
                f"{train_step}/{len(train_loader) // train_loader.batch_size}, "
                f"Train_loss: {train_loss.item():.4f}", end=" ")

            train_metric = dice_metric(outputs, label)
            epoch_metric_train += train_metric
            print(f'Train_dice: {train_metric:.4f}')

        print('-'*20)
        
        train_epoch_loss /= train_step
        print(f'Epoch_loss: {train_epoch_loss:.4f}')
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)
        
        epoch_metric_train /= train_step
        print(f'Epoch_metric: {epoch_metric_train:.4f}')

        save_metric_train.append(epoch_metric_train)
        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)

        if (epoch + 1) % test_interval == 0:

            model.eval()
            with torch.no_grad():
                test_epoch_loss = 0
                test_metric = 0
                epoch_metric_test = 0
                test_step = 0

                for test_data in test_loader:

                    test_step += 1
                    
                    test_volume = test_data["image"]
                    test_label = test_data["label"]
                    test_label = test_label != 0
                    if torch.cuda.is_available():
                      test_volume, test_label = (test_volume.to(device), test_label.to(device),)
                    
                    test_outputs = model(test_volume)
                    
                    test_loss = loss(outputs, test_label)
                    test_epoch_loss += test_loss.item()
                    test_metric = dice_metric(test_outputs, test_label)
                    epoch_metric_test += test_metric
                    
               
                test_epoch_loss /= test_step
                print(f'test_loss_epoch: {test_epoch_loss:.4f}')
                save_loss_test.append(test_epoch_loss)
                np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)

                epoch_metric_test /= test_step
                print(f'test_dice_epoch: {epoch_metric_test:.4f}')
                save_metric_test.append(epoch_metric_test)
                np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)

                if epoch_metric_test > best_metric:
                    best_metric = epoch_metric_test
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        model_dir, "best_metric_model.pth"))
                
                print(
                    f"current epoch: {epoch + 1} current mean dice: {test_metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )

            end = time.time()
            time_taken = end - start

            # record memory usage after running code
            gpu_memory_end = torch.cuda.max_memory_allocated(device=device)
            cpu_memory_end = psutil.Process().memory_info().rss

            # calculate memory usage
            max_gpu_memory = max(max_gpu_memory, round((gpu_memory_end - gpu_memory_start)// (1024*1024),2))
            max_cpu_memory = max(max_cpu_memory, round((cpu_memory_end - cpu_memory_start)// (1024*1024),2))
            
            print("Time Taken: ",time_taken)
            print("Maximum GPU Memory taken for training: ",max_gpu_memory)
            print("Maximum CPU Memory taken for training: ",max_cpu_memory)
            print(
                f"train completed, best_metric: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}")
            update_history([start_from, max_epochs, best_metric, best_metric_epoch, time_taken, max_cpu_memory, max_gpu_memory],model_dir=model_dir)


  