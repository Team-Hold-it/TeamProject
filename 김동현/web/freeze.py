from flask import Flask, render_template, send_from_directory
from flask_frozen import Freezer

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('1_index.html')

"""
# 다운로드 처리를 위한 함수
@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory('web_test_data', filename, as_attachment=True)


if __name__ == '__main__':
    freezer = Freezer(app)

    # URL 생성기 추가
    @freezer.register_generator
    def download_file():
        yield {'filename': 'example.txt'}

    freezer.freeze()
"""