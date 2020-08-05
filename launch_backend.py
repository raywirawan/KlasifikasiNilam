from flask import request, jsonify, Flask
from werkzeug.utils import secure_filename
import os
import uuid

from algorithm import start

#==================================================
#
# IF YOU ARE TESTING THE UPLOAD FEATURE
# PLEASE USE POSTMAN
# OR AN ANDROID APP
#
#==================================================
#
# TO TRAIN THE ALGORITHM WITH A NEW DATASET, PLEASE
# DELETE "training_features.csv" and "testing_features.csv"
# INSIDE THE "DATASET/csv" and "DATASET/training" FOLDER. 
# AFTER THAT, COPY THE NEW DATASET INTO THE FOLDER "DATASET/training". 
# THEN YOU CAN SEND A POST REQUEST TO /train 
# WITH A KEY NAMED "AUTHORIZE", WITH THE VALUE True
#
#==================================================
#
# CODE LIST:
#
#   SUCCESS:
#   0   : All okay!
#   1   : Training the "training" dataset succeeded  
#
#   ERRORS:
#   -1  : Upload failed, no image is selected
#   -2  : Upload failed, file is not supported
#   -3  : Training failed
#   -99 : Usually a fatal error. Please check the console logs!
#
#==================================================

#vars
UPLOAD_FOLDER       = "dataset/testing"
ALLOWED_EXTENSIONS  = [
    'png', 
    'jpg', 
    'jpeg'
    ]
FORM_KEY            = "image"

#populate string for logging purposes
STRING_OF_ALLOWED_EXTENSIONS = ""
for extension in ALLOWED_EXTENSIONS:
    STRING_OF_ALLOWED_EXTENSIONS += (extension + " ")

#init
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

#file extension check
def isExtensionAllowed(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#index
@app.route('/')
def hello():
    return "Hello."

@app.route('/jsontest')
def jsontest():
    return jsonify(
        status="Success",
        code=0,
        message="Also, hello. This is the expected json output from this software.",
        payload="Status is usually 'success' or 'failure'. Code is a simple way of referencing events, with a positive value meaning the program is running fine, while a negative value means there is something wrong. Message is usually an explanation of the json. The payload is usually the expected result of an operation. "
    )

#image training
@app.route('/train', methods = ['GET', 'POST'])
def handle_training():
    try:
        authorization = request.form['AUTHORIZE']
        if authorization:
            start.train()
            return jsonify(
                status="Success",
                code=1,
                message="Training has been completed",
                payload=""
            )
    except BaseException as e:
        return jsonify(
            status="Failed",
            code=-3,
            message="Training has failed.",
            payload=str(e)
        )

#all-in-one
@app.route('/predict', methods = ['GET', 'POST'])
def handle_prediction():
    try:
        imagefile = request.files[FORM_KEY]

        #check for empty image upload
        if imagefile == "":
            return jsonify(
                status="Upload failed",
                code=-1,
                message="No image is selected.",
                payload=""
            )

        filename = ""

        #check if file format is supported
        if imagefile and isExtensionAllowed(imagefile.filename):
            filename = secure_filename(imagefile.filename)
        else:
            return jsonify(
                status="Upload failed",
                code=-2,
                message="File not supported. Please make sure your image is format is: "+STRING_OF_ALLOWED_EXTENSIONS,
                payload=""
            )

        #logs to server
        print("\nReceived image File name : " + imagefile.filename)

        #generates a UUID for the uploaded file
        uuidstring = uuid.uuid4().hex

        #assigns the UUID as filename with the correct file extension
        filename = uuidstring+"."+filename.rsplit('.', 1)[1].lower()

        #saves the image in the server with the UUID as name
        imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        #extracts all the feature for 
        start.test(filename)

        #no parameter because it will take the last line of the csv
        #if this program is multithreaded (and multi-user perhaps) and something's gone wrong, please check this part
        prediction = start.predict()
        print(prediction)
        #returns the json
        return jsonify(
            status="Success",
            code=0,
            message="Image prediction procedure succeeded. ELM confidence:" + prediction[1],
            payload=prediction[0]
        )

    except BaseException as e:
        return jsonify(
            status="Fatal Error",
            code=-99,
            message="Image prediction failed: Cannot extract features from the image.",
            payload=str(e)
        )

if __name__ == '__main__':
    print("Starting up (native Flask)...")
    app.run(host='0.0.0.0', threaded=True)