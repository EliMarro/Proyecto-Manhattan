import os
from flask import Flask, render_template, request
from tensorflow import python
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import shutil
from shutil import copyfile
from datetime import date, datetime
app = Flask(__name__)
path = "C://Users//Álvaro//Desktop//Proyecto-Manhattan//Servidor//ImagenesUsuarios"
path2 = "C://Users//Álvaro//Desktop//Proyecto-Manhattan//Servidor//ImagenesANALizadas"
path3 = "C://Users//Álvaro//Desktop//Proyecto-Manhattan//Servidor"
app.config['UPLOAD_FOLDER'] = path

@app.route('/')
def home():
    return render_template('español.html')

@app.route("/", methods = ['POST'])
def uploader():
    #Se activa el modelo y se analiza
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
    import test
    x = test.predictImage()
    B = round(x[0],2)
    M = round(x[1],2)
    elementos =  os.listdir(path)
    for i in range(len(elementos)):
        if len(elementos) != 0 :
            P = os.path.splitext(elementos[i])
            print(P[0])
            date_m = (datetime.now().strftime('%d-%m-%Y'))
            m = str(date_m)
            #Codigo para cambiar las imagenes de carpeta
            origen = str(path + "//"+ elementos[i])
            destino = str(path2+"//"+ m +"-"+P[0]+".dcm") 
            shutil.move(origen,destino)
    os.remove("1-1.dcm")
    return render_template("enviar.html", B = B, M = M)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
