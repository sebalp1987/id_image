from imutils import contours
import numpy as np
import imutils
import cv2

def read_dni(img_file, size_w=599, size_h=407, angle = 0, shape_rect = (15,7), shape_sq = (5,5), gap_thres=0, gap_maxval=255, blur= None,
             adaptative=None, gamma=None, denoising=False):

    # construct the argument parser and parse the arguments
    image_color = cv2.imread(img_file)
    image = cv2.imread(img_file)

    image = imutils.rotate_bound(image, angle)
    image_color = imutils.rotate_bound(image_color, angle)

    image = imutils.resize(image, width=size_w, height=size_h)
    image_color = imutils.resize(image_color, width=size_w, height=size_h)

    if gamma != None:
        image = adjust_gamma(image, gamma=gamma)
        cv2.imwrite('gamma_adjust.jpg', image)

    if blur != None:
        image = cv2.medianBlur(image, blur)
        cv2.imwrite('blur.jpg', image)

    if denoising == True:
        image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
        cv2.imwrite('denoising.jpg', image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray.png', gray)

    # We define some processing-kernels
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, shape_rect)
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, shape_sq)

    # Invert background
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    cv2.imwrite('invert_back.jpg', tophat)

    # We isolate the digits
    gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
                      ksize=-1)

    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (500 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")
    cv2.imwrite('digits.jpg', gradX)


    # Close gaps between digits (twice to help)
    if adaptative == None:
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        thresh = cv2.threshold(gradX, gap_thres, gap_maxval,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        cv2.imwrite('close-digits1.png', thresh)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
        cv2.imwrite('close-digits2.png', thresh)

    else:
        if adaptative == 'gaussian':
            gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
            thresh = cv2.adaptiveThreshold(gradX,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
            cv2.imwrite('close-digits1.png', thresh)
        if adaptative == 'mean':
            gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
            thresh = cv2.adaptiveThreshold(gradX, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                        cv2.THRESH_BINARY, 11, 2)


    # Using the thresh we get the location
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    locs = []

    # Buscamos lo que queremos filtrar
    for (i, c) in enumerate(cnts):
        # compute the bounding box of the contour, then use the
        # bounding box coordinates to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)


        if (w > 500 and w < 600) and (h > 15 and h < 60):  # para esto ver el loc [(45, 310, 71, 22)]
                # append the bounding box region of the digits group
                # to our locations list
                locs.append((x, y, w, h))

    # Ordenamos los sitios que encontramos
    locs = sorted(locs, key=lambda x: x[0])
    output = []

    try:
        import Image
    except ImportError:
        from PIL import Image
        import PIL.ImageOps
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = 'C:/ZURICH/Tesseract-OCR/tesseract'
    tessdata_dir_config = '--tessdata-dir "C:\\ZURICH\\Tesseract-OCR\\tessdata"'

    for (i, (gX, gY, gW, gH)) in enumerate(locs):
        # initialize the list of group digits
        groupOutput = []
        try:
            # extract the group ROI of 4 digits from the grayscale image,
            # then apply thresholding to segment the digits from the
            # background of the credit card
            group = gray[gY - 5:gY + gH + 5, gX - 0:gX + gW + 5]
            name = 'group' + str(i) + '.png'

            group = cv2.threshold(group, 0, 255,
                                  cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            cv2.imwrite(name, group)

            img = Image.open(name)
            value = pytesseract.image_to_string(img, lang='spa', config=tessdata_dir_config)
            groupOutput.append(value)

        except:
            pass
        # draw the digit classifications around the group
        cv2.rectangle(image_color, (gX - 5, gY - 5),
                      (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
        cv2.putText(image_color, "".join(groupOutput), (gX, gY - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # update the output digits list
        output.extend(groupOutput)
        print(output)

        cv2.imshow("Image", image_color)
        cv2.imwrite('final-image.png', image_color)
        cv2.waitKey(0)
    import unidecode
    import re
    output = [re.sub(' ' + '+', '', i) for i in output]
    output = [re.sub('<' + '<' + '+', '<', i) for i in output]
    output = [unidecode.unidecode(i) for i in output]
    replace_elements = {'[':'I', '<':' ', "'":'', '|':''}
    for key, value in replace_elements.items():
        output = [i.replace(key, value) for i in output]

    # BIRTH = if the 6 first digits are mostly numbers is probabily the first sentence
    statistic = []
    for index, i in enumerate(output):

        birth = i[0:6]
        suma = sum(c.isdigit() for c in birth)
        statistic.append([suma, index])
    max_statistic = max(statistic)
    second_row = output[max_statistic[1]]
    second_row_det = re.sub(' ', '',second_row)
    day = second_row_det[4:6]
    month=second_row_det[2:4]
    year = ''
    if 50 <= int(second_row[0:2]) <= 99:
        year = '19' + str(second_row[0:2])
    if 0 <= int(second_row[0:2]) <= 25:
        year = '20' + str(second_row[0:2])

    print('NACIMIENTO :', str(day)+'-'+str(month)+'-'+str(year))

    # PAIS
    import pandas as pd
    pais = re.sub(r'[0-9]+', '', second_row)
    pais = pais[1:]
    lista = pd.read_csv('pais.csv', encoding='latin1', delimiter=';')
    lista = pd.Series(lista.COUNTRY.values, index=lista.ID).to_dict()
    for key, value in lista.items():
        if key in pais:
            pais = value
    print('NACIONALIDAD :', pais)
    del output[max_statistic[1]]

    # NAME
    statistic = []
    for index, i in enumerate(output):
        suma = sum(c.isalpha() for c in i)
        statistic.append([suma, index])
    max_statistic = max(statistic)
    third_row = output[max_statistic[1]]
    third_row = [i.upper() for i in third_row]
    replace_elements = {'0':'O', '1':'I', '5':'S','6':'G', '8':'B'}
    for key, value in replace_elements.items():
        third_row = [i.replace(key, value) for i in third_row]
    third_row = ''.join(third_row)
    print('NOMBRE: ', third_row)
    del output[max_statistic[1]]

    # NIF
    first_row = output[0].split(' ')
    first_row.remove('')
    statistic = []

    if len(first_row) > 1:
        first_row = ''.join(first_row)
        first_row = first_row[-9:]
        first_row =list(first_row)
        if first_row[-1].isdigit():
            first_row[-1] = replace_elements.get(first_row[-1])
        first_row = ''.join(first_row)
        print('NIE NUMBER: ', first_row)
    if len(first_row) == 1:
        first_row = first_row[0]
        first_row = first_row[-9:]

        first_row = list(first_row)
        if first_row[-1].isdigit():
            first_row[-1] = replace_elements.get(first_row[-1])
        first_row = ''.join(first_row)
        print('NIE NUMBER: ', first_row)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)



# read_dni('foto-dni4.jpg')
# read_dni('foto-dni5.jpg', denoising=True)
# read_dni('foto-dni6.jpg', size_w=599, size_h=407, angle= 270, shape_rect=(35,5), shape_sq=(5,5),gap_maxval=255, gap_thres=0)
# read_dni('foto-dni7.jpg', size_w=599, size_h=407, angle=269, shape_rect=(17,10), shape_sq=(15,10), gap_thres=10, gap_maxval=255,blur=None, adaptative=None, gamma=1.75)