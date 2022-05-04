import matplotlib.pyplot as plt
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from ResNET_keras_model import getAccuracyLoss_Train, getAccuracyLoss_Val
from ResNET_torch_model import getAccuracyLossTrain, getAccuracyLossVal

epochs = 30
batch_size = 64

epochs = range(1, epochs + 1, 1)



vgg_k_t_a, vgg_k_t_l = getAccuracyLoss_Train()

print('Keras Train')
print('Maximo valor de accuracy: {} Ultimo valor de accuracy: {}'.format(max(vgg_k_t_a), vgg_k_t_a[len(vgg_k_t_a) - 1]))
print('Menor valor de loss: {} Ultimo valor de loss: {}'.format(min(vgg_k_t_l), vgg_k_t_l[len(vgg_k_t_l) - 1]))


vgg_k_v_a, vgg_k_v_l = getAccuracyLoss_Val()

print('Keras Val')
print('Maximo valor de accuracy: {} Ultimo valor de accuracy: {}'.format(max(vgg_k_v_a), vgg_k_v_a[len(vgg_k_v_a) - 1]))
print('Menor valor de loss: {} Ultimo valor de loss: {}'.format(min(vgg_k_v_l), vgg_k_v_l[len(vgg_k_v_l) - 1]))


vgg_t_t_a, vgg_t_t_l = getAccuracyLossTrain()

print('Torch train')
print('Maximo valor de accuracy: {} Ultimo valor de accuracy: {}'.format(max(vgg_t_t_a), vgg_t_t_a[len(vgg_t_t_a) - 1]))
print('Menor valor de loss: {} Ultimo valor de loss: {}'.format(min(vgg_t_t_l), vgg_t_t_l[len(vgg_t_t_l) - 1]))

vgg_t_v_a, vgg_t_v_l = getAccuracyLossVal()


print('Torch Val')
print('Maximo valor de accuracy: {} Ultimo valor de accuracy: {}'.format(max(vgg_t_v_a), vgg_t_v_a[len(vgg_t_v_a) - 1]))
print('Menor valor de loss: {} Ultimo valor de loss: {}'.format(min(vgg_t_v_l), vgg_t_v_l[len(vgg_t_v_l) - 1]))



plt.plot ( epochs, vgg_k_t_a, 'r--', label='Training acc Keras'  )
plt.plot ( epochs, vgg_k_v_a,  'b', label='Validation acc Keras')
plt.plot ( epochs, vgg_t_t_a, 'g--', label='Training acc Torch'  )
plt.plot ( epochs, vgg_t_v_a,  'c', label='Validation acc Torch')
plt.title ('Training and validation accuracy Keras - Torch')
plt.ylabel('acc')
plt.xlabel('epochs')

plt.legend()
plt.figure()
plt.show()

plt.plot ( epochs, vgg_k_t_l, 'r--', label='Training loss Keras'  )
plt.plot ( epochs, vgg_k_v_l,  'b', label='Validation loss Keras')
plt.plot ( epochs, vgg_t_t_l, 'g--', label='Training loss Torch'  )
plt.plot ( epochs, vgg_t_v_l,  'c', label='Validation loss Torch')

plt.title ('Training and validation loss Keras - Torch')
plt.ylabel('loss')
plt.xlabel('epochs')

plt.legend()
plt.figure()
plt.show()