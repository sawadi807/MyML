import os
import shutil

# 원본 이미지가 있는 디렉토리 경로
src_dir = r"E:\대학_(2023)\졸논\archive\DermMel\train_sep\Melanoma"
# 복사될 이미지를 저장할 디렉토리 경로
dst_dir = r"E:\대학_(2023)\졸논\이미지데이터\Melanoma"

# 디렉토리가 없으면 생성합니다.
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

# 이미지 파일들의 경로를 리스트로 저장합니다.
img_files = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]

# 이미지 파일들을 복사합니다.
for f in img_files:
    src_file = os.path.join(src_dir, f)
    dst_file = os.path.join(dst_dir, f)
    shutil.copy2(src_file, dst_file)

print(len(os.listdir(dst_dir)))
