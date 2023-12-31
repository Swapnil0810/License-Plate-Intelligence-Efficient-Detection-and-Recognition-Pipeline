import cv2
from PIL import Image
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from ultralytics import YOLO
from paddleocr import PaddleOCR

ch_ppocr_mobile_cls = "resources/weights_onnx/cls_model.onnx"
ch_PP_OCRv3_det = "resources/weights_onnx/det_model.onnx"
ch_PP_OCRv3_rec = "resources/weights_onnx/rec_model.onnx"

ocr = PaddleOCR(use_angle_cls=True, lang="en", det_model_dir=ch_PP_OCRv3_det, rec_model_dir=ch_PP_OCRv3_rec, cls_model_dir=ch_ppocr_mobile_cls,use_onnx=True,use_gpu=True)
model = YOLO("resources/weights/best.pt")


def detector(image):
    result = model.predict(image, classes=0, verbose=False)[0].boxes.cpu().numpy()
    return result


def denoise_image(path):

    img = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)
    norm_img = np.zeros((img.shape[0], img.shape[1]))
    imag = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    converted_img = cv2.cvtColor(imag, cv2.COLOR_GRAY2BGR)
    deniosed = cv2.fastNlMeansDenoisingColored(converted_img, None, 10, 10, 7, 15)
    return deniosed


def check_extra_words(string):
    if string.isalpha() == True:
        if len(string) > 2:
            return False
        else:
            return str(string)
    else:
        return str(string)


def get_index_of_IND(list_result):
    position_x = dict()
    ind_index = []
    for idx, i in enumerate(list_result):
        position_x[idx] = int(i[0][0][0])
        check = i[-1][0]
        if "IND" == check or "I" == check or "N" == check or "D" == check or "IN" == check or "ND" == check or "ID" == check:
            ind_index.append(idx)
    min_position = min(position_x.values())
    key_of_min = [key for key in position_x if position_x[key] == min_position]
    for indx in key_of_min:
        if indx in ind_index:
            return indx


def get_distancewise_box(len, result):
    temp_dict = dict()
    origin = np.array((0.0, 0.0))  # giving origin coordinates
    for i in range(len):
        box_chord = result[0][i][0][0]  # getting coordinates of box
        point = np.array((box_chord[0], box_chord[1]))  # storing box coordinate
        dist = np.linalg.norm(origin - point)  # calculate the distance between box and origin
        temp_dict[int(dist)] = re.sub("[^A-Za-z0-9]+", "", str(result[0][i][-1][0])).upper()  # storing extracted text with their distance from origin
    myKeys = list(temp_dict.keys())
    myKeys.sort()

    sorted_dict = (str("".join([temp_dict[i] for i in myKeys if check_extra_words((str(temp_dict[i]).strip())) != False]))).strip()  # concate the extracted text sorted keywise in string
    return sorted_dict


def recognised_plate(img_path):

    new_path = denoise_image(img_path)
    result = ocr.ocr(new_path, det=True, cls=False)
    main_result = result[0]  ## inside of result

    if len(main_result) > 0:  ## check is there anything inside result

        score = 0
        event = str()
        for array in main_result:
            score += float(array[1][1])

        if score > 0.05 * (len(main_result)):
            event = "best"
        else:
            event = "next"

        if len(main_result) == 1:  ## if only one box is extracted
            plate = ""
            for part in main_result:
                temp_plate = check_extra_words((re.sub("[^A-Za-z0-9]+", "", str(part[-1][0]))).upper())
                if temp_plate == False:
                    pass
                else:
                    plate += temp_plate
            plate = plate.strip()
            return plate, event

        else:
            indx = get_index_of_IND(main_result)
            try:
                main_result.pop(indx)
            except:
                pass
            number = get_distancewise_box(len(main_result), result)
            return number, event


def number_plate_detection(frame, temp, mot_tracker, size=[150, 150]):
    bbox = detector(frame)
    result = []
    final_output = []

    Height, Width, Channel = frame.shape

    # boundries = crop.shape
    for cords in bbox:
        if cords.conf > 0.50:
            X, Y, W, H = cords.xywh[0]

            if ((int(X)-(W//2)) > 10) & ((int(Y)-(H//2)) > 10) & ((int(X) + int(W)) < int(Width - 10)) & ((int(Y) + int(H)) < int(Height - 10)):

                X, Y = X - (W // 2), Y - (H // 2)
                X, Y, W, H = int(X), int(Y), int(W), int(H)
                x1, y1, x2, y2 = X, Y, X + W, Y + H
                result.append((int(x1), int(y1), int(x2), int(y2)))

    if result:
        tracked_boxes = mot_tracker.update(np.array(result))
        if len(tracked_boxes):
            for bocses in range(len(tracked_boxes)):
                cropped_image = frame[int(tracked_boxes[bocses][1]) : int(tracked_boxes[bocses][3]), int(tracked_boxes[bocses][0]) : int(tracked_boxes[bocses][2])]
                try:
                    y_pred, event = recognised_plate(cropped_image)
                except:
                    y_pred, event = "not", "next"

                # for i in tracked_boxes:
                if int(tracked_boxes[bocses][4]) not in temp.keys() and event == "best":
                    temp[int(tracked_boxes[bocses][4])] = "best"
                    y_pred = str(y_pred.strip())
                    if len(y_pred) > 5 and len(y_pred) < 15:
                        # track = tracked_boxes[bocses]
                        return tracked_boxes[bocses] ,y_pred
                        # final_output.append([i, y_pred])

                if len(temp) > 500:
                    del temp[list(temp.keys())[0]]
                # return final_output
