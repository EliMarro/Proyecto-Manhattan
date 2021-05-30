import os
from flask import Flask, render_template, request
from tensorflow import python
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import shutil
from shutil import copyfile
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
    #Esto se prepara para hacer cosas NAZIS que solo Diana comprende(Referencia clara a su pasado nacional socialista)
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
    B = x[0]
    M = x[1]
    elementos =  os.listdir(path)
    for i in range(len(elementos)):
        if len(elementos) != 0 :
            #Codigo para cambiar las imagenes de carpeta
            origen = str(path + "//"+ elementos[i])
            destino = str(path2+"//"+elementos[i]) 
            shutil.move(origen,destino)
    os.remove("1-1.dcm")
    return render_template("enviar.html", B = B, M = M)

# @app.route('/resultados')        
# def resultados():
#     import test
#     x = test.predictImage()
#     B = x[0]
#     M = x[1]
#     print(B)
#     print(M)
#     # f = predictImage.__dict__
#     # print(f)

#     os.remove("1-1.dcm")
#     elementos =  os.listdir(path)
#     for i in range(len(elementos)):
#         if len(elementos) != 0 :
#             #Codigo para cambiar las imagenes de carpeta
#             origen = str(path + "//"+ elementos[i])
#             destino = str(path2+"//"+elementos[i]) 
#             shutil.move(origen,destino)
#     return render_template("resultados.html",B = B, M = M)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
