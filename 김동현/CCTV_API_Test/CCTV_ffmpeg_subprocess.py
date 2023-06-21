import pandas as pd
import subprocess
from tqdm import tqdm

# 지속 시간 설정 (초)
print("시간(초) 입력")
duration = int(input())

# stream_urls.csv 파일 불러오기
df_stream = ["http://210.179.218.53:1935/live/291.stream/playlist.m3u8"]
# df_stream = pd.read_csv('stream_urls.csv')

# 저장 폴더 경로
output_folder = 'D:/Study/TeamProject_2/CCTV_API_Test/web_test_data/'

stream_url = df_stream[0]
output_filename = "test_cctv.mp4"
command = f"ffmpeg -i \"{stream_url}\" -t {duration} -c copy \"{output_folder}{output_filename}\""
subprocess.run(command, shell=True)

# save_stream 함수 정의
def save_stream(stream_url, duration, output_filename):
    output_folder = 'D:/Study/TeamProject_2/CCTV_API_Test/web_test_data/'
    command = f"ffmpeg -i \"{stream_url}\" -t {duration} -c copy \"{output_folder}{output_filename}\""
    subprocess.run(command, shell=True)

"""
# URL 별로 반복하여 동영상 파일 저장
for i, row in tqdm(df_stream.iterrows(), desc='processing'):
    stream_url = row['StreamURL']
    output_filename = f"test_{i:02d}.mp4"
    command = f"ffmpeg -i \"{stream_url}\" -t {duration} -c copy \"{output_folder}{output_filename}\""
    subprocess.run(command, shell=True)
"""

print("동영상 파일 저장 완료")