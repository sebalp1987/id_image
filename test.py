from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plot


'------------------------TRAINING-------------------------------------------------------------------'
reference = cv2.imread('ocr-font-7.png', 0)
# reference = cv2.cvtColor(reference, cv2.COLOR_BAYER_BG2GRAY)
reference = cv2.threshold(reference, 50, 255, cv2.THRESH_BINARY_INV)[1]
ref_out = cv2.imwrite('test_tresh.png', reference)

refCnts = cv2.findContours(reference.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# loop over the OCR-A reference contours
for (i, c) in enumerate(refCnts):
	# compute the bounding box for the digit, extract it, and resize
	# it to a fixed size
	(x, y, w, h) = cv2.boundingRect(c)
	print(x,y,w,h)
	roi = reference[y:y + h, x:x + w]
	# roi = cv2.resize(roi, (9, 20))

	# update the digits dictionary, mapping the digit name to the ROI
	digits[i] = roi
	name = 'roi_digits' + str(i) + '.png'

'------------------------TEST-------------------------------------------------------------------------'

# construct the argument parser and parse the arguments
image_color = cv2.imread('foto-dni4.jpg')
image = cv2.imread('foto-dni4.jpg')
# Image resize
# gray = imutils.resize(image, width=1000)
gray = image
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.png', gray)

# We define some processing-kernels
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

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
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imwrite('close-digits1.png', thresh)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
image_out = cv2.imwrite('close-digits2.png', thresh)

'''
thresh = cv2.threshold(gray, 150, 300, cv2.THRESH_BINARY_INV)[1]
ref_out = cv2.imwrite('tresh.png', thresh)
'''
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
	ar = w / float(h)

	# Definimos lo que buscamos
	if ar > 2.5:
		# contours can further be pruned on minimum/maximum width
		# and height (hay que defnirlo por expermientacion)
		
		if (w > 500 and w < 600) and (h > 15 and h < 30): # para esto ver el loc [(45, 310, 71, 22)]
			# append the bounding box region of the digits group
			# to our locations list
			locs.append((x, y, w, h))

# Ordenamos los sitios que encontramos
locs = sorted(locs, key=lambda x:x[0])
output = []
print(locs)

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
		'''
		'---------------------------------------------------------------'
		group = cv2.threshold(group, 150, 300, cv2.THRESH_BINARY_INV)[1]
		ref_out = cv2.imwrite('group.png', group)
		
		'''
		print('tesseract')
		img = Image.open(name)
		value = pytesseract.image_to_string(img, lang='spa', config=tessdata_dir_config)
		print(value)
		groupOutput.append(value)

		'---------------------------------------------------------------'
		'''
		# detect the contours of each individual digit in the group,
		# then sort the digit contours from left to right

		digitCnts = cv2.findContours(group.copy(), cv2.RETR_TREE,
									 cv2.CHAIN_APPROX_NONE)
		digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]
		digitCnts = contours.sort_contours(digitCnts,
										   method="left-to-right")[0]

		
		name = str(i) +'.png'
		cv2.imwrite(name, group)
		img = Image.open(name)
		print(pytesseract.image_to_string(img, lang='spa', config=tessdata_dir_config))
	

		# loop over the digit contours
		i=0
		for c in digitCnts:

			i = i + 1

			# compute the bounding box of the individual digit, extract
			# the digit, and resize it to have the same fixed size as
			# the reference OCR-A images
			(x, y, w, h) = cv2.boundingRect(c)
			roi = group[y:y + h, x:x + w]
			roi = cv2.resize(roi, (9, 20))
			name = 'roi_' + str(i) + '.png'
			cv2.imwrite(name, roi)

			# initialize a list of template matching scores
			scores = []
			files = []

			# loop over the reference digit name and digit ROI
			for (digit, digitROI) in digits.items():
				# apply correlation-based template matching, take the
				# score, and update the scores list
				name = 'digitROI' + str(digit) + '.png'
				cv2.imwrite(name, digitROI)
				result = cv2.matchTemplate(roi, digitROI,
										   cv2.TM_CCOEFF)
				(_, score, _, _) = cv2.minMaxLoc(result)
				scores.append(score)
				files.append(name)
			print(files)
			print(scores)
			# the classification for the digit ROI will be the reference
			# digit name with the *largest* template matching score
			groupOutput.append(str(np.argmax(scores)))
			print(groupOutput)
	
		scores = []
		file_open =['']
		for (digit, digitROI) in digits.items():
			print(digit)
			name_ROI = 'digit_ROI' + str(digit) + '.png'
			print(name_ROI)
			cv2.imwrite(name_ROI, digitROI)
			cv2.imwrite('roi.png', group)
			result = cv2.matchTemplate(group, digitROI, cv2.TM_CCOEFF)
			cv2.imshow('result', result)
			(_, score, _, _) = cv2.minMaxLoc(result)
			print(name_ROI, score)
			scores.append(score)
	
		groupOutput.append(str(np.argmax(scores)))
	
		print(groupOutput)
		'''

	except:
		pass
	# draw the digit classifications around the group
	cv2.rectangle(image_color, (gX - 5, gY - 5),
				  (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
	cv2.putText(image_color, "".join(groupOutput), (gX, gY - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

	# update the output digits list
	output.extend(groupOutput)

	print("DNI NUMBER #: {}".format("".join(output)))
	cv2.imshow("Image", image_color)
	cv2.imwrite('final-image.png', image_color)
	cv2.waitKey(0)


