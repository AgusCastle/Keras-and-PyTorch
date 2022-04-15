import matplotlib.pyplot as plt
from deep_net_keras import Net_Keras
from deep_net_torch import Net_Torch

epocas = 15
input_size = 28 * 28
output_size = 10
batch_size = 128
lr = 0.001

epochs = range(1, epocas + 1, 1)

# Keras
net_keras = Net_Keras(epocas)

net_keras.modelNet()
net_keras.trainAndVal()

# PyTorch

net_torch = Net_Torch(input_size, output_size, batch_size, lr, epocas)

net_torch.train()
net_torch.eval()

# Keras vectores

acc_t_k, loss_t_k = net_keras.getAccuracyLoss_Train()
acc_v_k, loos_v_k = net_keras.getAccuracyLoss_Val()

# Torch vectores

acc_t_t, loss_t_t = net_torch.getAccuracyLoss_Train()
acc_v_t, loss_v_t = net_torch.getAccuracyLoss_Val()

# Pasamos a la graficacion accuracy

plt.plot ( epochs, acc_t_k, 'r--', label='Training acc Keras'  )
plt.plot ( epochs, acc_v_k,  'b', label='Validation acc Keras')

plt.plot ( epochs, acc_t_t, 'k--', label='Training acc Torch'  )
plt.plot ( epochs, acc_v_t,  'g', label='Validation acc Torch')

plt.title ('Training and validation accuracy Keras - Torch')
plt.ylabel('acc')
plt.xlabel('epochs')

plt.legend()
plt.figure()

# Pasamos a la graficacion loss

plt.plot ( epochs, loss_t_k, 'r--', label='Training loss Keras'  )
plt.plot ( epochs, loos_v_k,  'b', label='Validation loss Keras')

plt.plot ( epochs, loss_t_t, 'k--', label='Training loss Torch'  )
plt.plot ( epochs, loss_v_t,  'g', label='Validation loss Torch')

plt.title ('Training and validation loss Keras - Torch')
plt.ylabel('acc')
plt.xlabel('epochs')

plt.legend()
plt.figure()
