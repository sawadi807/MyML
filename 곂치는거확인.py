# import os

# dir1 = r"G:\내 드라이브\졸논\practice_train\malignant"
# dir2 = r"E:\대학_(2023)\졸논\archive\DermMel\valid\Melanoma"

# # 각 디렉토리에서 이미지 파일을 찾기 위해 파일 이름을 리스트로 저장합니다.
# img_files1 = [f for f in os.listdir(dir1) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
# img_files2 = [f for f in os.listdir(dir2) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

# # 각 디렉토리의 파일 개수를 구합니다.
# file_count1 = len(os.listdir(dir1))
# file_count2 = len(os.listdir(dir2))

# # 두 리스트에서 곂치는 이미지 파일의 개수를 구합니다.
# overlap_count = len(set(img_files1).intersection(img_files2))

# print("두 디렉토리에서 곂치는 이미지 파일의 개수:", overlap_count)
# print("dir1에 있는 이미지 파일 개수:", len(img_files1), ", 전체 파일 개수:", file_count1)
# print("dir2에 있는 이미지 파일 개수:", len(img_files2), ", 전체 파일 개수:", file_count2)
