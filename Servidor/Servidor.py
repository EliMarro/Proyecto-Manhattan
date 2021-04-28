import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./ImagenesUsuarios"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/CNN.html')
def red():
    return render_template('CNN.html')

@app.route('/formulario.html')
def imagenes():
    return render_template('formulario.html')

@app.route("/uploader", methods = ['POST'])
def uploader():
    if request.method == "POST":
        f = request.files["Imagen"]
        print(f)
        # if f>=1:
        #     for files in f:
        #         filename = secure_filename(f.filename)
        #         files.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return "Archivo subido exitosamente"+filename


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
