"""
    @author: breadcrumbbuilds
    based on read_multispectral_cli.py by @franama

"""
import sys
import numpy as np
import os.path as path
import matplotlib.pyplot as plt
import cv2
import time as t

rawImagePath = "../images/raw/"
convertImagePath = "../images/converted/"
image = rawImagePath + "sentinel2"
binaryImage = image + ".bin"
headerImage = image + ".hdr"

# Parses an images header file. Returns the header's values


def getCharacteristicValues(headerFile):
    characteristics = {"samples": 0, "lines": 0, "bands": 0}
    for line in open(headerFile).readlines():
        for characteristic in characteristics.keys():
            if characteristic in line:
                words = line.split('=')
                value = words[1].strip()
                characteristics[characteristic] = value
    return characteristics


def convertBinary():
    characteristicValuesDict = getCharacteristicValues(headerImage)
    bands = int(characteristicValuesDict["bands"])
    lines = int(characteristicValuesDict["lines"])
    samples = int(characteristicValuesDict["samples"])
    print("Bands", bands, "\nLines", lines, "\nSamples", samples)

    data = np.fromfile(binaryImage, '<f4').reshape((bands,  lines * samples))
    print("Bytes Read: " + str(data.size))
    return data, bands, lines, samples


def createRGB(data, bands, lines, samples):
    band_select = [3, 2, 1]
    rgb = np.zeros((samples, lines, 3))
    for i in range(0, 3):
        rgb[:, :, i] = data[band_select[i], :].reshape((samples, lines))

    return rgb


def showImage(data):
    plt.imshow(data)
    plt.tight_layout()
    plt.show()
    plt.close()


"""
    Writes a png image to the converted folder
"""


def uniqueTimeString():
    time = t.localtime()
    timeStr = str(time[0]) + '-' + str(time[1]) + '-' + str(time[2]) + \
        '-' + str(time[3]) + str(time[4]) + str(time[5])
    return timeStr


def writeImage(data):
    timeStr = uniqueTimeString()
    imagePath = "../images/converted/" + timeStr + ".png"
    status = cv2.imwrite(imagePath, data*255)
    if status:
        print("Image", str(imagePath), "written")
    else:
        print("Error, image was not written")


def run():
    d, b, l, s = convertBinary()
    rgb = createRGB(d, b, l, s)

    showImage(rgb)
    writeImage(rgb)


run()
