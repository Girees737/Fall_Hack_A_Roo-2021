from flask import redirect, url_for
import Load_Model as lm
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
app = Flask(__name__)

app.config['UPLOAD_FOLDER']='/Images/'

# app.config['MAX_CONTENT_PATH']=

@app.route('/')
def render_static():
    return render_template('/image_load.html')

# when the post method detect, then redirect to success function
@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        filenam=f.filename
        print(filenam)
        # return "File uploaded successfully"
        output=lm.predict(filenam)
        # redirect(url_for('success', name=f))
        return render_template('render_image.html')

if __name__ == '__main__':
    app.run("192.168.0.20",debug=False,port=5003, threaded=True)