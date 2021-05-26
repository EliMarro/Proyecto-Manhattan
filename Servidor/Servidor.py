import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import shutil
import random
app = Flask(__name__)
path = "C://Users//Álvaro//Desktop//Proyecto-Manhattan//Servidor//ImagenesUsuarios"
path2 = "C://Users//Álvaro//Desktop//Proyecto-Manhattan//Servidor//ImagenesANALizadas"
app.config['UPLOAD_FOLDER'] = path

@app.route('/')
def home():
    return render_template('español.html')

@app.route("/", methods = ['POST'])
def uploader():
    #Esto se prepara para hacer cosas NAZIS que solo Diana comprende(Referencia clara a su pasado nacional socialista)
    #Se activa el modelo y se analiza
    resultado = random.random()
    accuaracy = resultado
    if request.method == "POST":
        f = request.files.getlist("imagen")
        for i in range(len(f)):
            x = f[i].filename
            filename = secure_filename(x)
            f[i].save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
    return render_template("enviar.html", accuaracy = resultado)

@app.route('/background_process_test')        
def background_process_test():
    elementos =  os.listdir(path)
    for i in range(len(elementos)):
        if len(elementos) != 0 :
            print("Hay imagenes que analizar")
            #Aqui iria la parte de activar el modelo
            print("LA IMAGEN"+elementos[i]+"SE VAN A ANALIZAR")
            #Codigo para cambiar las imagenes de carpeta
            origen = str(path + "//"+ elementos[i])
            destino = str(path2+"//"+elementos[i]) 
            shutil.move(origen,destino)
    return ("Nothing")


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
