import os
import shutil

# 검증-정상
dir1 = r"E:\NotMelanoma_Gray"
dir2 = r"E:\대학_(2023)\졸논\최종데이터셋\검증\정상"


files1 = os.listdir(dir1)
for i in range(3000):
    file1 = files1[i]
    src1 = os.path.join(dir1, file1)
    dst1 = os.path.join(dir2, file1)
    shutil.move(src1, dst1)

# 검증-흑색종
dir3 = r"E:\Melanoma_Augmented"
dir4 = r"E:\대학_(2023)\졸논\최종데이터셋\검증\흑색종"

files2 = os.listdir(dir3)
for i in range(3000):
    file2 = files2[i]
    src2 = os.path.join(dir3, file2)
    dst2 = os.path.join(dir4, file2)
    shutil.move(src2, dst2)


# 훈련-정상
dir5 = r"E:\NotMelanoma_Gray"
dir6 = r"E:\대학_(2023)\졸논\최종데이터셋\학습\정상"

files3 = os.listdir(dir5)
for i in range(12000):
    file3 = files3[i]
    src3 = os.path.join(dir5, file3)
    dst3 = os.path.join(dir6, file3)
    shutil.move(src3, dst3)


# 훈련-흑색종
dir7 = r"E:\Melanoma_Augmented"
dir8 = r"E:\대학_(2023)\졸논\최종데이터셋\학습\흑색종"

files4 = os.listdir(dir7)
for i in range(12000):
    file4 = files4[i]
    src4 = os.path.join(dir7, file4)
    dst4 = os.path.join(dir8, file4)
    shutil.move(src4, dst4)


# 테스트-정상
dir9 = r"E:\NotMelanoma_Gray"
dir10 = r"E:\대학_(2023)\졸논\최종데이터셋\테스트\정상"

files5 = os.listdir(dir9)
for i in range(341):
    file5 = files5[i]
    src5 = os.path.join(dir9, file5)
    dst5 = os.path.join(dir10, file5)
    shutil.move(src5, dst5)


# 테스트-흑색종
dir11 = r"E:\Melanoma_Augmented"
dir12 = r"E:\대학_(2023)\졸논\최종데이터셋\테스트\흑색종"

files6 = os.listdir(dir11)
for i in range(1023):
    file6 = files6[i]
    src6 = os.path.join(dir11, file6)
    dst6 = os.path.join(dir12, file6)
    shutil.move(src6, dst6)
