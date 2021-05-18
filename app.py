from flask import Flask, redirect, url_for, request, render_template
from tensorflow.keras.models import load_model
import pickle
import cv2
import numpy as np



app=Flask(__name__)
model = pickle.load(open('mango_jackfruit.pkl', 'rb'))
#model = load_model('my_model')


@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    img=request.files['image']
    img.save('img.jpg')
    image=cv2.imread("img.jpg")
    image = cv2.resize(image, (224,224))
    image=image/255
    image = np.expand_dims(image, axis=0)
 
    preds = model.predict(image)
    if preds[0][0] == 1:
        preds="mango"
    else:
        preds="jackfruit"


        
    return render_template('index.html',prediction_text=preds)













if __name__=='__main__':
    app.run(debug=True)