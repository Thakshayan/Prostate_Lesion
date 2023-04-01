import matplotlib.pyplot as plt
import numpy as np
import os

def load_metrices(path):
  metrices_dir = path
  #if n : metrices_dir = OUT_DIR + 'tries/try' + str(n) + "/"
  train_loss = np.load(os.path.join(metrices_dir, 'loss_train.npy'))
  train_metric = np.load(os.path.join(metrices_dir, 'metric_train.npy'))
  test_loss = np.load(os.path.join(metrices_dir, 'loss_test.npy'))
  test_metric = np.load(os.path.join(metrices_dir, 'metric_test.npy'))
  return train_loss, train_metric, test_loss, test_metric


def plt_metrices(path):
  train_loss, train_metric, test_loss, test_metric= load_metrices(path)

  plt.figure("Results 25 june", (12, 6))
  plt.subplot(2, 2, 1)
  plt.title("Train dice loss")
  x = [i + 1 for i in range(len(train_loss))]
  y = train_loss
  plt.xlabel("epoch")
  plt.plot(x, y)

  plt.subplot(2, 2, 2)
  plt.title("Train metric DICE")
  x = [i + 1 for i in range(len(train_metric))]
  y = train_metric
  plt.xlabel("epoch")
  plt.plot(x, y)

  plt.subplot(2, 2, 3)
  plt.title("Test dice loss")
  x = [i + 1 for i in range(len(test_loss))]
  y = test_loss
  plt.xlabel("epoch")
  plt.plot(x, y)

  plt.subplot(2, 2, 4)
  plt.title("Test metric DICE")
  x = [i + 1 for i in range(len(test_metric))]
  y = test_metric
  plt.xlabel("epoch")
  plt.plot(x, y)
  plt.savefig(path+'/metrics.png')
  plt.show()