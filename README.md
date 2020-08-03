# SDRD MODELO API

## Para instalar las librerias
```
pip install -r requirements.txt
```

## Para iniciar
```
python app.py
```

## USO

### Modelos

OBTENER MODELOS DISPONIBLES->Petición GET
http://127.0.0.1:5000/modelos

CARGAR MODELO->Petición GET
http://127.0.0.1:5000/<modelo>
http://127.0.0.1:5000/resnet18ft

### Diagnostico

OBTENER DIAGNOSTICO->Petición POST
http://127.0.0.1:5000/diagnostico
body.files['imagen'] # imagen
body.form['nombre'] # nombre del modelo (opcional)
