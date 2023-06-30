# CCTV 스트림 페이지
import pandas as pd
import subprocess, os

# cctv url불러오기
df = pd.read_csv("CCTV_API_Test/CCTVID_URL.csv", index_col=0)

# CCTV.py
def get_all_cctv_names():
    return df['CCTVNAME'].tolist()

# Get stream URL by CCTV name
def get_stream_url(cctv_name):
    return df[df['CCTVNAME'] == cctv_name]['StreamURL'].values[0]

# CCTV.py
def save_stream(cctv_name, duration):
    stream_url = get_stream_url(cctv_name)
    output_filename = f"{cctv_name}_.mp4"
    # output_folder = '/mnt/d/Study/web_holdit/web_holdit_local/downloads/' # 로컬 작업시 경로
    output_folder = '/app/web/static/downloads/' # 앱 배포시 경로
    output_path = os.path.join(output_folder, output_filename)  # 전체 경로를 미리 생성
    command = f"ffmpeg -y -i \"{stream_url}\" -t {duration} -c copy \"{output_path}\""
    result = subprocess.run(command, shell=True)

    if result.returncode != 0:  # ffmpeg 명령이 성공적으로 수행되지 않았다면
        # 빈 mp4 파일 생성
        error_filename = f"해당 지역에서 현재 스트리밍 영상을 제공하지 않습니다_{cctv_name}.mp4"
        error_output_path = os.path.join(output_folder, error_filename)
        command = f"ffmpeg -y -f lavfi -i anullsrc -t 0.1 -c:a aac \"{error_output_path}\""
        subprocess.run(command, shell=True)
        return error_filename  # 에러가 발생했음을 알리는 파일 이름 반환
    else:
        return output_filename  # 정상적으로 저장된 파일 이름 반환

# 이미지 캡쳐 함수
def capture_frames(video_path, output_folder, frame_rate):
    # ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # output format - we'll use the video filename, and append the timestamp to it
    output_format = os.path.join(output_folder, os.path.basename(video_path).split('.')[0] + '_%04d.jpg')
    command = f"ffmpeg -i {video_path} -vf fps={frame_rate} {output_format}"
    subprocess.run(command, shell=True)
