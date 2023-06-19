from flask import Flask, render_template, request
import os, sys
from werkzeug.utils import secure_filename
import cv2

# 결로 설정
path = os.getcwd() # C:\Users\user\section6\tp2\code_file
yolo_path = path + '/yolov5/'
img_path = path + '/web/static/images/img/'
predict_path = path + '/web/static/images/'

app = Flask(__name__)
# /predict에서 업로드 폴더 설정할 때 쓰임
app.config['UPLOAD_FOLDER'] = './web/static/images/img'

# main page
@app.route('/')
def index():
    return render_template('1_index.html'), 200

# project introduce page
@app.route('/pjintro')
def pjintro():
    return render_template('2_pjintro.html'), 200

# model introduce page
@app.route('/mdintro')
def mdintro():
    return render_template('3_mdintro.html'), 200

# model development page
@app.route('/develop')
def develop():
    return render_template('4_develop.html'), 200

# service page
@app.route('/service')
def service():
    return render_template('5_service.html'), 200

# remind page
@app.route('/remind')
def remind():
    return render_template('6_remind.html'), 200

# service-predict page
@app.route('/predict', methods=['GET','POST'])
def detect():
    if request.method == 'POST':
        # 오류방지를 위해 미리 설정
        img_file = None
        predict_file = None

        # /service 페이지에서 받아온 파일
        img = request.files['img']
        filename = secure_filename(img.filename)

        # 확장자 확인 (only jpg of mp4)
        input_extension = filename.split('.')[-1]

        if input_extension == 'jpg' or input_extension == 'mp4':
            # 이미지 저장
            img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # 터미널 실행해서 예측
            terminnal_cmd = f'python {yolo_path}detect.py --source {img_path}{filename} --weights {yolo_path}runs/best.pt --project {predict_path} --name predict --img 640 --conf 0.3 --exist-ok'
            os.system(terminnal_cmd)

            # static 폴더안에 경로 설정
            img_file = 'images/img/' + filename
            predict_file = 'images/predict/' + filename

        return render_template('7_predict.html',
                               img_file=img_file,
                               predict_file=predict_file,
                               input_extension=input_extension)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)