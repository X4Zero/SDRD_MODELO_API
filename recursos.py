# Recursos.py

## Librerías
from __future__ import print_function, division
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image
import time
import os
# import copy

## Asignacion del dispositivo
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sigmaX=30
IMG_SIZE = 224

# DiccClases = {
#     1 : "Normal",
#     2 : "RD_Moderate",
#     3 : "RD_Heavy"
# }

DiccClases = {
    1 : "Ojo sano",
    2 : "Retinopatía Diabética no proliferativa leve o moderada",
    3 : "Retinopatía Diabética no proliferativa severa o Retinopatía Diabética Proliferativa"
}


## Funciones para el preprocesamiento

def cv_loader(path):
  return cv2.imread(path)

# Función para el recorte de las imágenes
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

## Preprocesamiento
def Preprocesamiento(impath, sigmaX=30,ispath=True):
    if ispath:
        image = cv2.imread(impath)
    else:
        image = impath

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , sigmaX) ,-4 ,128)
    im_pil = Image.fromarray(image)
    image = np.array(im_pil)
    image = image/255.
                       
    mean = np.array([0.485, 0.456, 0.406])
    sd = np.array([0.229, 0.224, 0.225])
                       
    image = ((image - mean) / sd)
    
    image = np.transpose(image, (2, 0, 1))
    
    return image


def Predict(image_path, model,device, topk=3,ispath=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    # Modelo al dispositivo
    model.to(device)
    model.eval()
    
    # Preprocesamiento
    img_torch = Preprocesamiento(image_path,30,ispath)
    img_torch = torch.from_numpy(img_torch)
    
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.to(device)) # use cuda
        
    probability = F.softmax(output.data,dim=1) # use F

    probs = probability.topk(topk)[0][0].cpu().numpy()
    index_to_class = {val: key for key, val in model.class_to_idx.items()} # from reviewer advice
    top_classes = [np.int(index_to_class[each]) for each in probability.topk(topk)[1][0].cpu().numpy()]
    
    return probs, top_classes

def DiagnosticoRD(image_path,modelo,ispath=True): 
    start = time.time()
    probs, classes = Predict(image_path, modelo,device,ispath=ispath)
    max_index = classes[0] 

    probabilidades = probs
    clases = [DiccClases[c] for c in classes]

    print ('tiempo')
    tiempo = time.time() - start
    print (tiempo)
    print ('Diagnostico')
    # clases = list(map(int, clases))
    probabilidades = list(map(float, probabilidades))
    print(probs)
    print(classes)
    print(clases)

    return {"diagnostico":DiccClases[max_index],"clases":clases,"probabilidades":probabilidades}

## encontrar el modelo

def encontrar_modelos(root_path):
  '''
    Función que busca archivos de modelos a partir de root_path, que es la ruta base

    args:
      root_path: ruta base

    returns:
      lista con las rutas de todos los modelos encontrados
  '''
  # carpetas
  folders = os.listdir(root_path)
  # creamos las rutaas
  rutas = [ os.path.join(root_path,path) for path in folders]

  modelos_directorios = []
  # revisamos dentro de cada carpeta
  for rt in rutas:
    if os.path.isdir(rt):
      # Si es directorio, buscamos dentro
      path_dir = rt
      archivos_dir = os.listdir(path_dir)
      for archivo in archivos_dir:
        if '.pth' in archivo: # buscamos los modelos y los guardamos
          ruta_modelo = os.path.join(path_dir,archivo)
          modelos_directorios.append(ruta_modelo)
  if len(modelos_directorios)> 0:
    print("Se encontraron modelos")
  else:
      print("No se encontraron modelos")
  return modelos_directorios

## nombres modelos
def obtener_modelos_nombres(path_source,sep="/"):
    rutas_modelos = encontrar_modelos(path_source)
    # Obtenemos las arquitecturas de los modelos
    modelos_disponibles = {ruta_modelo.split(sep)[-2].lower():ruta_modelo for ruta_modelo in rutas_modelos }
    print(modelos_disponibles)
    return modelos_disponibles

## modelos obtenidos
def obtener_modelos(path_source,device,sep="/",num=0):
    rutas_modelos = encontrar_modelos(path_source)
    # Obtenemos las arquitecturas de los modelos
    nombres_modelos = [ruta_modelo.split(sep)[-2].lower() for ruta_modelo in rutas_modelos]
    # Diccionario que contiene los modelo
    diccionario_modelos = {}
    diccionario_modelos_datos = {}
    print("Cargando modelos...")
    cont = 0
    for ruta, nombre in zip(rutas_modelos, nombres_modelos):
      if cont==num and cont !=0:
        break
      else:
        
        modelo_rec, diccionario_rec, best_accuracy, next_epoch = cargar_modelo_ckp(ruta, device)

        diccionario_modelos[nombre] = modelo_rec
        diccionario_modelos_datos[nombre] = {
            "best_accuracy":best_accuracy,
            "num_epochs":next_epoch - 1,
            "diccionario":diccionario_rec
        }

      cont+=1

    # Mostramos los modelos cargados
    print("\nSe cargaron los siguientes modelos: ")
    for modelo_cargado in diccionario_modelos.keys():
      print("\t-{}".format(modelo_cargado))

    return diccionario_modelos, diccionario_modelos_datos


## Cargado del modelo

def cargar_modelo_ckp(filepath,device):

    checkpoint = torch.load(filepath,map_location=torch.device(device))
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    if checkpoint['arch'][:6] == 'resnet':
      model.fc = checkpoint['classifier']
    else:
      model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    diccionario_reccuperado = checkpoint['metricas']
    best_accuracy = checkpoint['best_accuracy']
    next_epoch = checkpoint['next_epoch']

    return model, diccionario_reccuperado, best_accuracy, next_epoch

def cargar_modelo_nombre(nombre,path_source,sep):
    modelos_disponibles = obtener_modelos_nombres(path_source,sep)

    path = modelos_disponibles[nombre]

    modelo_rec, diccionario_rec, best_accuracy, next_epoch = cargar_modelo_ckp(path, device)
    
    return modelo_rec




# revisamos que modelos hay
# base_path = os.path.join(os.getcwd(),'modelos')

# rutas_modelos = encontrar_modelos(base_path)
# print(rutas_modelos)

# cargamos los modelos
# modelos_recuperados, modelos_recuperados_detalle = obtener_modelos(base_path,device,sep="\\",num=1)
# print(modelos_recuperados.keys())