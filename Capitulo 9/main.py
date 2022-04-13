from deep_net_keras import Net_Keras

net_keras = Net_Keras()

net_keras.modelNet()
net_keras.trainAndVal()

net_keras.getModel() # Supongo que al obtener podemos obtener los datos para generar las graficas


'''
Aqui en el main se traeran los datos de keras y pytorch para generar las graficas
obtendremos de 

Keras
    Accuracy Train
    Accuracy Val
PyTorch
    Accuracy Train
    Accuracy Val

Con en el eje de las x con la epoca
Asi mismo tambien evaluaremos la perdida

Keras
    Loss Train
    Loss Val
PyTorch
    Loss Train
    Loss Val

'''