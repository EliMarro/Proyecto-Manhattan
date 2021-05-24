import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = r"C:\Users\Álvaro\Desktop\Proyecto-Manhattan\Servidor\ImagenesUsuarios"

@app.route('/')
def home():
    return render_template('español.html')

@app.route("/", methods = ['POST'])
def uploader():
    if request.method == "POST":
        f = request.files.getlist("imagen")
        for i in range(len(f)):
            x = f[i].filename
            filename = secure_filename(x)
            f[i].save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
            return render_template("enviar.html")

@app.route('/background_process_test')        
def background_process_test():
    
    print("Analizando su imagen")
    return("Nothing")


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
