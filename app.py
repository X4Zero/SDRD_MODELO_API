from flask import Flask
from flask import jsonify
from flask import request

import numpy
import threading
import time
from recursos import *
app = Flask(__name__)

modelos_recuperados = {}

def cargar_modelo_inicial():
    with app.app_context():
        global modelos_recuperados

        modelo='resnet18ft'
        base_path = os.path.join(os.getcwd(),'modelos')
        modelo_obtenido = cargar_modelo_nombre(modelo,base_path,sep="\\")
        modelos_recuperados[modelo] = modelo_obtenido
        print('Se cargó el primer modelo')

@app.before_first_request
def activate_job():
    thread = threading.Thread(target=cargar_modelo_inicial)
    thread.start()

@app.route('/')
def index():
    global modelos_recuperados
    base_path = os.path.join(os.getcwd(),'modelos')

    modelos_disponibles = obtener_modelos_nombres(base_path,"\\")
    print(modelos_disponibles)
    return 'API - MODELO'

@app.route('/modelos')
def obtener_modelos_disponibles():
    global modelos_recuperados
    
    if len(modelos_recuperados.keys())<1:
        print("sleep")
        time.sleep(2)

    base_path = os.path.join(os.getcwd(),'modelos')

    modelos_disponibles = obtener_modelos_nombres(base_path,"\\")

    return jsonify({"modelos_disponibles_sistema":list(modelos_recuperados.keys()),
    "modelos_disponibles":list(modelos_disponibles)})

@app.route('/modelos/<modelo>')
def cargar_modelo(modelo):
    global modelos_recuperados

    base_path = os.path.join(os.getcwd(),'modelos')

    #revisamos los modelos disponibles
    modelos_disponibles = obtener_modelos_nombres(base_path,"\\")

    if modelo in list(modelos_disponibles.keys()):
        modelo_obtenido = cargar_modelo_nombre(modelo,base_path,sep="\\")
        modelos_recuperados[modelo] = modelo_obtenido

        return obtener_modelos_disponibles()
    else:
        return jsonify({"mensaje":"no se ha encontrado el modelo {} en el sistema".format(modelo)})



@app.route('/diagnostico',methods=['POST'])
def obtener_diagnostico():
    if request.method == 'POST':
        global modelos_recuperados

        if len(modelos_recuperados.keys())<1:
            print("sleep")
            time.sleep(2)
            
        print(request.files)
        print(request.form)
        img = request.files['imagen']

        # por defecto tomamos el primero
        nombre_modelo=list(modelos_recuperados.keys())[0]
        if bool(request.form):
            nombre_modelo = request.form['nombre']

        print(nombre_modelo)
        if nombre_modelo in list(modelos_recuperados.keys()):
            img = cv2.imdecode(numpy.fromstring(img.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)

            modelo_usar = modelos_recuperados[nombre_modelo]
            diagnostico = DiagnosticoRD(img, modelo_usar,ispath=False)
            print(diagnostico)
            return jsonify(diagnostico)
        else:
            return jsonify({"mensaje":"no se ha encontrado el modelo {} en el sistema".format(nombre_modelo)})


if __name__ == '__main__':
    app.run(debug=True,port=5000)