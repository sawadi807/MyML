"""
추론을 거쳐서 benign 또는 malignant로 분류하여 저장한 np array인 output_result와
정답을 np array인 test_label을 비교하여 정확도를 계산하여 return
"""
import numpy as np


def evaluation_for_SC(output_results, test_label):
    prediction_digit = []
    label_digit = []

    for output_result in output_results:
        output_index = np.argmax(output_result)
        prediction_digit.append(output_index)

    prediction_digit = np.array(prediction_digit)
    label_digit = np.array(test_label)
    accuracy = (prediction_digit == label_digit).mean()
    return accuracy
