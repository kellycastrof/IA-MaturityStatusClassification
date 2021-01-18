  
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import os
from pathlib import Path

root = os.path.abspath(os.path.dirname(__file__))
tipos=["Class A/", "Class B/", "Class C/", "Class D/"]


modeloMB='./modelo/modeloMB.h5'
pesosMB='./modelo/pesosMB.h5'

modelMB=load_model(modeloMB)
modelMB.load_weights(pesosMB)



def analizar(imagen):
    x=load_img(imagen,target_size=(224,224))
    x=img_to_array(x)
    x=np.expand_dims(x,axis=0)
    x=x/255
    arreglo=modelMB.predict(x)
    resultado=arreglo[0]
    maximo= np.amax(resultado)
    if maximo < 0.1:
        return "None"
    respuesta=np.argmax(resultado)
    return tipos[respuesta]

def test():    
    for i in tipos:
        path = Path(root +'/datasetreal/test/'+i)
        total=0
        aciertos=0
        for entry in os.listdir(path):
            file_path=os.path.join(path, entry)
            if os.path.isfile(file_path):
                result = analizar(file_path)
                if result==tipos[tipos.index(i)]:
                    aciertos=aciertos+1
                total=total+1
        print("clase: %s aciertos: %d totales: %d "%(i,aciertos, total))


#test()