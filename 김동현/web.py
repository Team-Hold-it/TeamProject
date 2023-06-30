from flask import Flask, render_template, request, send_file, session, url_for, flash, redirect
from zipfile import ZipFile
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import logging, time, os, shutil, cv2

app = Flask(__name__, template_folder='templates')
app.logger.setLevel(logging.INFO)
app.config['UPLOAD_FOLDER'] = './static/images/img/' # 로컬 경로 "/mnt/d/Study/web_holdit/web_holdit_local/static/images/img/"
app.config['DOWNLOAD_FOLDER'] = './web/static/downloads' # 로컬 경로 "/mnt/d/Study/web_holdit/web_holdit_local/static/images/img/"
app.config['CAPTURE_FOLDER'] = './static/capture/' # 로컬 경로 "/mnt/d/Study/web_holdit/web_holdit_local/static/images/img/"
app.config['SECRET_KEY'] = 'your-secret-key' # save_stream 기능에서 필요. 실제 코드에서는 'your-secret-key' 대신 안전한 키를 사용해야 합니다.

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

# 객체 탐지
from PIL import Image

# 파일 확장자 검사
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'mp4'}
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# 경로 설정
v8_predict_path = "/app/runs/detect" # 로컬 경로 "/mnt/d/Study/web_holdit/web_holdit_local/runs/detect/"

# 입력 이미지 리사이즈
def resize_image(image_path, size=640):
    image = cv2.imread(image_path)

    height, width, _ = image.shape

    if max(height, width) != size:
        scale = size / max(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale)

    return image

# 객체 탐지
@app.route('/service/predict', methods=['POST'])
def predict():
    if 'img' not in request.files:
        return 'No file part', 400
    file = request.files['img']
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        # filename = secure_filename(file.filename)
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        app.logger.info(f'File successfully saved to {filepath}')

        input_extension = filename.split('.')[-1]
        if input_extension == 'jpg' or input_extension == 'mp4': 
            
            # YOLOv8 예측
            # YOLOv8 예측 전에 이미지 리사이즈
            # resized_image = resize_image(filepath)
            # cv2.imwrite(filepath, resized_image)  # 리사이즈된 이미지를 원래 파일 경로에 다시 저장
            
            model = YOLO("./best.pt") # 로컬 경로 "/mnt/d/Study/web_holdit/web_holdit_local/best.pt"
            results = model.predict(source=filepath, conf=0.25, save=True, exist_ok=True) # v8_predict_path 에 저장됨
            
            # predict 함수에 저장 경로 수정 파라미터가 없어서 따로 코드로 옮기기
            predict_directory = request.form.get('predict_directory', 'predict')  # YOLOv8 결과 저장 디렉터리 이름을 request form에서 가져오거나 기본값으로 'predict' 설정
            source_path = os.path.join(v8_predict_path, predict_directory, filename)
            target_path = os.path.join("./static/images/predict/", filename) # 로컬 경로 /mnt/d/Study/web_holdit/web_holdit_local/static/images/predict/

            app.logger.info(f"Moving file from {source_path} to {target_path}")
            app.logger.info(f"{source_path} 에 있는 파일 확인: {os.path.isfile(source_path)}")  # 소스에 파일이 있는지 로그로 남기기
            app.logger.info(f"{target_path} 에 있는 파일 확인: {os.path.isfile(target_path)}")  # 타겟에 파일이 있는지 로그로 남기기

            shutil.move(source_path, target_path)
            
            app.logger.info(f"이동 후 {source_path} 에 있는 파일 확인: {os.path.isfile(source_path)}")  # 이동 후 소스에 파일이 있는지 로그로 남기기
            app.logger.info(f"이동 후 {target_path} 에 있는 파일 확인: {os.path.isfile(target_path)}")  # 이동 후 타겟에 파일이 있는지 로그로 남기기

            if os.path.isfile(target_path):
                app.logger.info(f"File successfully moved to {target_path}")
            else:
                app.logger.info(f"File not found at {target_path}")

            # results 타입 확인 및 수정
            if isinstance(results, list) and len(results) > 0:
                results = results[0]
            
            # 정확한 이미지 경로를 얻기 위해 Flask의 `url_for` 함수를 사용합니다.
            img_url = url_for('static', filename='images/img/' + filename)  # 로컬 경로 "/static/images/img/" + filename
            predict_url = url_for('static', filename='images/predict/' + filename) # 로컬 경로 "/static/images/predict/" + filename

            # 클래스 이름 매핑
            class_names = {
                0: 'normal',
                1: 'pedestrian',
                2: 'helmet',
                3: 'Jaywalking',
                4: 'signal',
                5: 'stopline',
            }

            # 위반 사항 정보 추출
            violations = []

            for bbox in results.boxes:
                class_id = int(bbox.cls[0])  # 클래스 ID는 bbox의 'cls' 속성에 위치합니다.
                class_name = class_names[class_id] if class_id < len(class_names) else 'unknown'
                violation = f'Violation: {class_name}'
                violations.append(violation)

            time.sleep(2)

            return render_template('prediction.html', img_file=img_url, predict_file=predict_url, violations=violations), 200

# CCTV 스트림 페이지
from CCTV import capture_frames, save_stream, get_all_cctv_names

# '/cctv' 경로를 처리하는 함수
@app.route('/cctv', methods=['GET', 'POST'])  # 이 경로는 GET 및 POST 메서드를 모두 처리
def cctv():
    cctvnames = get_all_cctv_names()  # 모든 CCTV 이름을 가져옴
    selected_cctvnames = []  # 선택된 CCTV 이름들을 저장할 리스트
    saved_files = []  # 저장된 파일 이름을 저장할 리스트
    # downloads_dir = "/mnt/d/Study/web_holdit/web_holdit_local/downloads/" # 로컬 경로
    downloads_dir = app.config['DOWNLOAD_FOLDER'] # 웹 경로

    if request.method == 'POST':  # 요청 메서드가 POST인 경우
        selected_cctvnames = request.form.getlist('cctvname')  # 선택한 CCTV 이름 받아오기
        time = int(request.form['time'])  # form 데이터에서 시간 값을 가져옴 (정수로 변환)
        
        for cctv_name in selected_cctvnames:
            saved_file = os.path.join(downloads_dir, save_stream(cctv_name, time))
            if "해당 지역에서 현재 스트리밍 영상을 제공하지 않습니다" in saved_file: # 녹화 실패시 
                return '현재 CCTV 제공이 되지 않는 위치가 포함될 수 있습니다. 녹화에 실패한 CCTV는 빈 파일로 생성됩니다.', 500
            saved_files.append(saved_file)
            print(f'Recording Time : {selected_cctvnames}', time)
        
        session['saved_files'] = saved_files  # 저장된 파일 이름을 세션에 저장
          
        # 'cctv.html' 템플릿을 렌더링하여 응답으로 반환. CCTV 이름과 시간을 템플릿에 전달
        return render_template('cctv.html', cctvnames=cctvnames, selected_cctvname=selected_cctvnames), 200
    else:  # 요청 메서드가 GET인 경우
        cctvnames = get_all_cctv_names()  # 모든 CCTV 이름을 가져옴
        
        # 'cctv.html' 템플릿을 렌더링하여 응답으로 반환. CCTV 이름 목록을 템플릿에 전달
        return render_template('cctv.html', cctvnames=cctvnames), 200

# CCTV 다운로드 함수
@app.route('/download', methods=['GET'])
def download():
    print("Download route accessed")
    zip_path = "cctv_recordings.zip"
    
    with ZipFile(zip_path, 'w') as zip_file:
        for filename in session.get('saved_files', []):
            zip_file.write(filename, arcname=os.path.basename(filename))  # zip 파일에 추가

    # 압축 후 파일 삭제
    for filename in session.get('saved_files', []):
        if os.path.exists(filename):
            os.remove(filename)

    return send_file(zip_path, mimetype='application/zip', as_attachment=True, download_name='cctv_recordings.zip')

from flask import Flask, send_file
import os
from zipfile import ZipFile

# 이미지 캡쳐
@app.route('/capture', methods=['GET', 'POST'])
def capture():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part in the request.', 400
        file = request.files['file']

        # if user does not select file, file variable will be empty
        if file.filename == '':
            return 'No file selected for uploading.', 400

        # save file to the disk
        filename = secure_filename(file.filename)
        saved_path = os.path.join('/tmp', filename)
        file.save(saved_path)

        # capture frames
        output_folder = app.config['CAPTURE_FOLDER']
        capture_frames(saved_path, output_folder, frame_rate=1)  # capture one frame every second
        
        flash('동영상 캡처가 완료되었습니다.')
        return redirect(url_for('capture'))
    
    else:  # GET request
        return render_template('capture.html') 

# 이미지 다운로드 함수
@app.route('/capture_download', methods=['GET'])
def capture_download():
    print("Download route accessed")
    zip_path = "capture_images.zip"
    images_folder = app.config['CAPTURE_FOLDER']

    with ZipFile(zip_path, 'w') as zip_file:
        for filename in os.listdir(images_folder):
            file_path = os.path.join(images_folder, filename)
            zip_file.write(file_path, arcname=os.path.basename(file_path))  # zip 파일에 추가

    # 압축 후 파일 삭제
    for filename in os.listdir(images_folder):
        file_path = os.path.join(images_folder, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    return send_file(zip_path, mimetype='application/zip', as_attachment=True, download_name='capture_images.zip')

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    app.run(debug=False, host='0.0.0.0', port=port)