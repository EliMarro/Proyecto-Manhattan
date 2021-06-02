import os
from flask import Flask, render_template, request
from tensorflow import python
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import shutil
from shutil import copyfile
from datetime import date, datetime
app = Flask(__name__)#Se inicia el servidor
#Tres paths necesarios para el correcto funcionamiento del servidor
path = "C://Users//Álvaro//Desktop//Proyecto-Manhattan//Servidor//ImagenesUsuarios"
path2 = "C://Users//Álvaro//Desktop//Proyecto-Manhattan//Servidor//ImagenesAnalizadas"
path3 = "C://Users//Álvaro//Desktop//Proyecto-Manhattan//Servidor"
app.config['UPLOAD_FOLDER'] = path
#Se definene las dos URL que va a usar el servidor
@app.route('/')#Ruta principal
def home():
    return render_template('español.html')

@app.route("/upload", methods = ['POST'])#Ruta para subir una imagen
#Función de gestión de imagenes
def uploader():
    #Bucle para subir las imagenes
    if request.method == "POST":
        f = request.files.getlist("imagen")
        for i in range(len(f)):
            x = f[i].filename
            y = str(x)
            filename = secure_filename(y)
            origen = str(path+"//"+y)
            destino = str(path+"//"+"1-1.dcm")
            servidor = str(path3+"//"+"1-1.dcm")
            f[i].save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            os.rename(origen,destino)
            copyfile(destino,servidor)
    #Activación del modelo para el analisis de la imagen
    import test
    x = test.predictImage()
    B = round(x[0],2)
    M = round(x[1],2)
    #Bucle para gestionar las imagenes una vez han sido analizadas
    elementos =  os.listdir(path)
    for i in range(len(elementos)):
        if len(elementos) != 0 :
            P = os.path.splitext(elementos[i])
            print(P[0])
            date_m = (datetime.now().strftime('%H-%M-%S'))
            m = str(date_m)
            #Codigo para cambiar las imagenes de carpeta
            origen = str(path + "//"+ elementos[i])
            destino = str(path2+"//"+ m +"-"+P[0]+".dcm") 
            shutil.move(origen,destino)
    os.remove("1-1.dcm")
    return render_template("enviar.html", B = B, M = M)
#Runeo del servidor e indicación de los puertos
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
