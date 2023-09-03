import cv2
import numpy as np

# 이미지 읽기
img = cv2.imread(r'E:\aug_adapthisteq_ISIC_0030055.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 경계 검출
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# 모폴로지 연산 적용
kernel = np.ones((3,3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)


# 결과 이미지 출력
cv2.imshow('result', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()