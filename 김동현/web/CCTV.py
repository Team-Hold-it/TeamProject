# CCTV 스트림 페이지
import pandas as pd
import subprocess

# cctv url불러오기
df = pd.read_csv("D:\Study\TeamProject_2\CCTV_API_Test\CCTVID_URL", index_col=0)

# CCTV.py
def get_all_cctv_names():
    return df['CCTVNAME'].tolist()

# Get stream URL by CCTV name
def get_stream_url(cctv_name):
    return df[df['CCTVNAME'] == cctv_name]['StreamURL'].values[0]

# save_stream 함수 정의
def save_stream(cctv_name, duration, output_filename):
    stream_url = get_stream_url(cctv_name)
    output_folder = 'D:/Study/TeamProject_2/CCTV_API_Test/web_test_data/'
    command = f"ffmpeg -i \"{stream_url}\" -t {duration} -c copy \"{output_folder}{output_filename}\""
    subprocess.run(command, shell=True)