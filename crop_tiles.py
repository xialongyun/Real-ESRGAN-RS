import os
import cv2 as cv
import numpy as np


def combine(inputPath,startx,endx,starty,endy,dest):
    width=(endx-startx)*1024
    height=(endy-starty)*1024
    img = np.zeros((height, width, 3), dtype="uint8")
    folder = os.path.dirname(inputPath)
    for i in range(startx, endx, 1):
        for j in range(endy, starty, -1):
            tw=(i-startx)*1024
            th=(endy-j)*1024
            # tilePath = inputPath + str(i) + "/" + str(j) + ".webp"
            tilePath2 = inputPath +"png_output/"+ str(i) + "/" + str(j) + "_out.png"
            print(tilePath2)
            tile=cv.imread(tilePath2)
            output_dir = os.path.dirname(tilePath2)
            # os.makedirs(output_dir, exist_ok=True)
            # cv.imwrite(tilePath2, tile)
            img[th:th+1024,tw:tw+1024]=tile
    print(dest)
    output_dir = os.path.dirname(dest)
    os.makedirs(output_dir, exist_ok=True)
    cv.imwrite(dest, img)
    return

combine('./data/test_0407_tiles/', 222365, 222404, 166792, 166822, './data/test_0407_tiles/output_png.png')