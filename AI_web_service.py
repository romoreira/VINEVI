from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import os
from flask import Flask
from flask import request
from load_example import cnn_predict
from PIL import Image
import json


UPLOAD_FOLDER = '/home/ubuntu/api'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


@app.route('/aioracle', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        print("Size of received files: " + str(request.files))
        if 'image' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['image']
        packet_name = str(file1.filename)
        file1 = Image.open(file1)
        #print("Teste name2: "+str(file1))
        #path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        #file1.save(path)
        result = cnn_predict(file1)
        dictionary = {'packet':packet_name, 'predicted_class': result}
        json_result = json.dumps(dictionary,  indent=2)
        #print("Result: "+str(result) + " File Name: "+str(packet_name))
        return json_result


#        return 'ok'
#    return '''
#    <h1>Upload new File</h1>
#    <form method="post" enctype="multipart/form-data">
#      <input type="file" name="file1">
#      <input type="submit">
#    </form>
#    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)