from flask import Flask, render_template, request, redirect
from model import Model
import cv2, base64
from PIL import Image
import io
from flask import *

app = Flask(__name__, template_folder = 'templates')
app.config['SECRET_KEY'] = "shilu"
model = Model()
@app.route('/', methods = ['GET', 'POST'])
def home() :
    if request.method == 'POST' :
        img = request.files['my_image']
        img_path = "newimages/" + img.filename	
        img.save(img_path)
        img = cv2.imread("newimages/" + img.filename)
        model = Model()
        answer = model.run(img)
        print(answer)
        if not answer :
          flash("Sorry!!Try Someother Picture Currently model cannot classify it...", "error")
          return redirect(url_for("home"))
        else :          
          return render_template('result.html', result = answer[1])
    else :
        return render_template('home.html')
    
if __name__ == "__main__" :
    app.run(debug = True, port = 3000)
