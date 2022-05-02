import matplotlib.pyplot as plt
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from vgg_keras_model import VGGKeras
from vgg_torch_model import VggNetTorch

epochs = 30
batch_size = 64

vgg_keras = VGGKeras(epochs, 64)
vgg_keras.trainAndVal()


vgg_torch = VggNetTorch(0.001, epochs, device)
vgg_torch.train()




vgg_t_t_a, vgg_t_t_l = vgg_torch.getAccuracyLossTrain()
vgg_t_v_a, vgg_t_v_l = vgg_torch.getAccuracyLossVal()

vgg_k_t_a, vgg_k_t_l = vgg_keras.getAccuracyLoss_Train()
vgg_k_v_a, vgg_k_v_l = vgg_keras.getAccuracyLoss_Val()

plt.plot ( epochs, vgg_k_t_a, 'r--', label='Training acc Keras'  )
plt.plot ( epochs, vgg_k_v_a,  'b', label='Validation acc Keras')
plt.plot ( epochs, vgg_t_t_a, 'r--', label='Training acc Torch'  )
plt.plot ( epochs, vgg_t_v_a,  'b', label='Validation acc Torch')
plt.title ('Training and validation accuracy Keras - Torch')
plt.ylabel('acc')
plt.xlabel('epochs')

plt.legend()
plt.figure()
plt.show()

plt.plot ( epochs, vgg_k_t_l, 'r--', label='Training loss Keras'  )
plt.plot ( epochs, vgg_k_v_l,  'b', label='Validation loss Keras')
plt.plot ( epochs, vgg_t_t_l, 'r--', label='Training loss Torch'  )
plt.plot ( epochs, vgg_t_v_l,  'b', label='Validation loss Torch')

plt.title ('Training and validation loss Keras - Torch')
plt.ylabel('loss')
plt.xlabel('epochs')

plt.legend()
plt.figure()
plt.show()