import matplotlib.pyplot as plt

from vgg_keras_model import VGGKeras

epochs = 250
batch_size = 64

vgg_keras = VGGKeras(250, 64)

vgg_keras.trainAndVal()

vgg_k_t_a, vgg_k_t_l = vgg_keras.getAccuracyLoss_Train()
vgg_k_v_a, vgg_k_v_l = vgg_keras.getAccuracyLoss_Val()

plt.plot ( epochs, vgg_k_t_a, 'r--', label='Training acc Keras'  )
plt.plot ( epochs, vgg_k_v_a,  'b', label='Validation acc Keras')

plt.title ('Training and validation accuracy Keras - Torch')
plt.ylabel('acc')
plt.xlabel('epochs')

plt.legend()
plt.figure()
plt.show()

plt.plot ( epochs, vgg_k_t_l, 'r--', label='Training loss Keras'  )
plt.plot ( epochs, vgg_k_v_l,  'b', label='Validation loss Keras')

plt.title ('Training and validation loss Keras - Torch')
plt.ylabel('loss')
plt.xlabel('epochs')

plt.legend()
plt.figure()
plt.show()