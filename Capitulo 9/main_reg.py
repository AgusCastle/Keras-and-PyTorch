import matplotlib.pyplot as plt
from regression_model_keras import RegressionModelKeras
from regression_model_torch import RegressionModelTorch

epocas = 32
input_size = 13
output_size = 1
batch_size = 8
lr = 0.001

epochs = range(1, epocas + 1, 1)

# Modelo de keras

keras = RegressionModelKeras()

keras.trainAndVal()

# Modelo de Torch

ptorch = RegressionModelTorch(input_size, output_size, batch_size, lr, epocas)

ptorch.train()

loss_t_k = keras.getLoss_Train()
loss_v_k = keras.getLoss_Val()

loss_t_t = ptorch.getTrain_Loss()
loss_v_t = ptorch.getVal_Loss()


plt.plot ( epochs, loss_t_k, 'r--', label='Training loss Keras'  )
plt.plot ( epochs, loss_v_k,  'b', label='Validation loss Keras')

plt.plot ( epochs, loss_t_t, 'k--', label='Training loss Torch'  )
plt.plot ( epochs, loss_v_t,  'g', label='Validation loss Torch')

plt.title ('Training and validation loss Keras - Torch')
plt.ylabel('loss')
plt.xlabel('epochs')

plt.legend()
plt.figure()