from flask import Flask, render_template
from flask import request
from flask_cors import *
from model import text_predict
import datetime
import os
import json
import time
import requests

app = Flask(__name__)
CORS(app, resources=r'/*')

phone_num_dict = {}

@app.route('/')
def hello_world():
    '''返回首页'''
    return render_template('index.html')

# @app.route('/search')
# def search():
#     content = request.args.get('search')
#     vec = compute_sentence_vec(content)
#     dict = compute_cosine_sim(vec)
#     return render_template('index.html', dict=dict)

@app.route('/search_photo', methods=['POST'])
def search_photo():
    nowTime = datetime.datetime.now()
    file = request.files['file']
    filename = file.filename
    dst = os.path.dirname(__file__) + '/tmp/' + str(nowTime) + '.jpg'
    file.save(dst)
    content = text_predict(dst)
    os.remove(dst)
    return content

# @app.route('/file')
# def file():
#     return render_template('file.html')

@app.route('/test',methods=['GET','POST'])
def test():
    data = json.loads(request.form.get('data'))
    imgname = data['imgname']
    pleacehold= data['pleacehold']
    print(imgname)
    print(pleacehold)
    return "img/小人/2.jpg"




if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8081)
