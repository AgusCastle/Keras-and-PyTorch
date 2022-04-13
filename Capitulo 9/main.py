import matplotlib.pyplot as plt
from deep_net_keras import Net_Keras

#       (** Eje X **) (** Eje Y **) Esta solo es una prueba 
# -- Debemos pasar todos los resultados 
# a listas o tuplas para poder generar la grafica

# Este es un ejemplo nadamas
fig, ax = plt.subplots()
dias = ['L', 'M', 'X', 'J', 'V', 'S', 'D'] # Estas serian nuestras epocas
temperaturas = {'Madrid':[28.5, 30.5, 31, 30, 28, 27.5, 30.5], # valores de accuracy
                'Barcelona':[24.5, 25.5, 26.5, 25, 26.5, 24.5, 25]} # Valores loss
#
ax.plot(dias, temperaturas['Madrid'], linestyle = 'dashed')
ax.plot(dias, temperaturas['Barcelona'], linestyle = 'dotted')
plt.show()

# Este ejemplo ya tenemos todo

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