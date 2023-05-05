import base64
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import imutils
import builtins
from collections import Counter
import cv2
import numpy as np
import time
import os

def getCoordinates(data):

    # final_coord

  base64_string = data

  base64_string = base64_string.replace('data:image/png;base64,', '')

    # Add padding to the Base64 string if necessary
  missing_padding = len(base64_string) % 4


    # Decode the Base64 string
  decoded_data = base64.b64decode(base64_string)

    # Create a PIL image from the decoded data
  img = Image.open(BytesIO(decoded_data))

    # Show the image
  img.show()
  img.save('s.png')
  # Load the image
  image1 = cv2.imread('s.png')  # Replace 'image.jpg' with the path to your image

  # Convert the image to HSV color space
  hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)

  # Define the red color range in HSV format
  lower_red1 = np.array([0, 10, 10])
  upper_red1 = np.array([10, 255, 255])
  lower_red02 = np.array([150, 10, 10])
  upper_red02 = np.array([180, 255, 255])

  # Threshold the image to get only red color
  mask1 = cv2.inRange(hsv1, lower_red1, upper_red1)
  mask2 = cv2.inRange(hsv1, lower_red02, upper_red02)
  mask12 = cv2.bitwise_or(mask1, mask2)
  
  
  

  # Find contours in the mask
  contours1, _ = cv2.findContours(mask12, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Extract the region surrounding the redline and add 10 pixels padding
  for contour in contours1:
      x, y, w, h = cv2.boundingRect(contour)
      region = image1[max(0, y - 10):min(y + h + 10, image1.shape[0] - 1), max(0, x - 10):min(x + w + 10, image1.shape[1] - 1)]

  # Save the extracted region as a new image
  cv2.imwrite('s1.png', region)  # Replace 'redline_region.jpg' with the desired output image filename and format


  def get_start_end_cord():
      
    # Load the image
    img = cv2.imread('s.png')
    print("hi")

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for red color in HSV
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Threshold the image to get only red color
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours of the red color
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_1, y_1 = 0, 0
    # Loop over the contours and check if they are a line
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        if len(approx) == 2:
            # Get the coordinates of the line
            x_1, y_1 = approx[0][0]
            x_2, y_2 = approx[1][0]
            
    print(x_1)
    return x_1, y_1



  def get_image_orientation(file_path):
      # Open the image file
      with Image.open(file_path) as im:
          # Get the width and height
          width, height = im.size
          # Determine orientation based on aspect ratio
          if width > height:
              return 'horizontal'
          else:
              return 'vertical'

  startcord,endcord=get_start_end_cord()
  print(startcord)
  print(endcord)
  time.sleep(3)
  type1=get_image_orientation('s1.png')
  temVar='hort'
  if(type1 == 'vertical'):
    temVar='vert'
    image = Image.open("s1.png")

    rotated_image = image.transpose(method=Image.ROTATE_90)

    rotated_image.save('s1.png')

  img = cv2.imread(r"s1.png")

  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  
  (T, thresh)  = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
  
  kernel = np.ones((3, 3), np.uint8)
    
  invert = cv2.bitwise_not(thresh)

  dilation = cv2.dilate(invert, kernel, iterations=3)





  cv2.imwrite("image-after-remove-small-lines.jpg", dilation)
  img = cv2.imread('image-after-remove-small-lines.jpg')

  # Convert the image to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Threshold the image to create a binary mask of the black regions
  _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

  # Define a kernel for morphological operations
  kernel = np.ones((10,1), np.uint8)

  # Apply erosion to the binary mask in the horizontal direction to increase the thickness of the black lines
  eroded_mask = cv2.erode(mask, kernel, iterations=3)

  # Apply the eroded mask to the original image to isolate the thickened black lines
  result = cv2.bitwise_and(img, img, mask=eroded_mask)
  cv2.imwrite("image-after-remove-small-lines.jpg", result)



    
  img = cv2.imread('image-after-remove-small-lines.jpg')
  imgLines = img.copy()
  imgGaps = img.copy()

  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  (T, thresh)  = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

  #find all lines in the image
  lines = cv2.HoughLinesP(thresh,1,np.pi/180,25)

  for line in lines:
      
      for x1,y1,x2,y2 in line:
          #Draw lines on image
          cv2.line(imgLines,(x1,y1), (x2,y2), (0,255,0),1)
  #display the image
  cv2.imwrite("test.png", imgLines)
  cv2.waitKey(0)




  img = cv2.imread('./image-after-remove-small-lines.jpg')
  imgLines = img.copy()
  imgGaps = img.copy()

  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  (T, thresh)  = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

  #find all lines in the image
  lines = cv2.HoughLinesP(thresh,1,np.pi/180,15)

  for line in lines:
      
      for x1,y1,x2,y2 in line:
          #Draw lines on image
          cv2.line(imgLines,(x1,y1), (x2,y2), (0,255,0),1)
  #display the image
  cv2.imwrite("test.png", imgLines)
  cv2.waitKey(0)




  def order_points_old(pts):
      rect = np.zeros((4, 2), dtype="float32")
      
      s = pts.sum(axis=1)
      rect[1] = pts[np.argmin(s)]
      rect[0] = pts[np.argmax(s)]

      diff = np.diff(pts, axis=1)
      rect[2] = pts[np.argmin(diff)]
      rect[3] = pts[np.argmax(diff)]

      return rect




  def order_points(pts):
      xSorted = pts[np.argsort(pts[:, 0]), :]
      
      leftMost = xSorted[:2, :]
      rightMost = xSorted[2:, :]

      leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
      (tl, bl) = leftMost
      
      D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
      (br, tr) = rightMost[np.argsort(D)[::-1], :]
      
      return np.array([tl, tr, br, bl], dtype="float32")



  ap = argparse.ArgumentParser()
  ap.add_argument("-n", "--new", type=int, default=-1, 
                  help="whether or not the new order points should be used")


  image = cv2.imread(r'image-after-remove-small-lines.jpg')
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (7, 7), 0)





  edged = cv2.Canny(gray, 500, 80)
  edged = cv2.dilate(edged, None, iterations=1)
  edged = cv2.erode(edged, None, iterations=1)


  cv2.waitKey(0)
  cv2.destroyAllWindows()



  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] 




  (cnts, _) = contours.sort_contours(cnts)
  colors = ((0, 0, 255), (240, 0, 150), (255, 0, 0), (255, 255, 0))


  cords_save=[]


  for (i, c) in enumerate(cnts):
      
          
      box = cv2.minAreaRect(c)
      box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
      box = np.array(box, dtype="int")
      cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
      # show the original coordinates
      cords_save.append(box)
      
      rect = order_points_old(box)

      for ((x, y), color) in zip(rect, colors):
          cv2.circle(image, (int(x), int(y)), 5, color, -1)
          
      cv2.putText(image, "Object #{}".format(i + 1), (int(rect[0][0] - 15), int(rect[0][1] - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
      cv2.waitKey(0)
      cv2.destroyAllWindows()





  def midpoint(ptA, ptB):
      return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

  # construct the argument parse and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--image", required=True, help="path to the input image")
  ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in inches)")


  edged = cv2.Canny(gray, 50, 100)
  ref_start_point = (3,5)
  ref_end_point = (280,5)
  edged = cv2.dilate(edged, None, iterations=1)
  edged = cv2.erode(edged, None, iterations=1)

  # find contours in the edge map
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] 

  # sort the contours from left-to-right and initialize the
  # 'pixels per metric' calibration variable
  (cnts, _) = contours.sort_contours(cnts)
  pixelsPerMetric = None

  cnts_list=[]
  for c in cnts:
      # if the contour is not sufficiently large, ignore it

  # compute the rotated bounding box of the contour
      orig = image.copy()
      box = cv2.minAreaRect(c)
      box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
      box = np.array(box, dtype="int")
  # order the points in the contour such that they appear
  # in top-left, top-right, bottom-right, and bottom-left
  # order, then draw the outline of the rotated bounding
  # box
      box = perspective.order_points(box)
      cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
  # loop over the original points and draw them
      for (x, y) in box:
          cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
          
          (tl, tr, br, bl) = box
      (tltrX, tltrY) = midpoint(tl, tr)
      (blbrX, blbrY) = midpoint(bl, br)
      
      cnts_list.append([tl, tr])

  # compute the midpoint between the top-left and top-right points,

  # followed by the midpoint between the top-righ and bottom-right
      (tlblX, tlblY) = midpoint(tl, bl)
      (trbrX, trbrY) = midpoint(tr, br)
  # draw the midpoints on the image
      cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
      cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
      cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
      cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
  # draw lines between the midpoints
      cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
      cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)
      dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
      dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
      if pixelsPerMetric is None:
          pixelsPerMetric = dB / 0.955
          
      dimA = dA / pixelsPerMetric
      dimB = dB / pixelsPerMetric
      cv2.putText(orig, "{:.1f}in".format(dimA), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 2)
      cv2.putText(orig, "{:.1f}in".format(dimB), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 2)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

  def midpoint(ptA, ptB):
      return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)



  image = cv2.imread(r'image-after-remove-small-lines.jpg')
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (7, 7), 0)
  cv2.line(image, (25,5), (68,5), (255,0,0), 2)
  cv2.line(image, (99,5), (139,5), (255,0,0), 2)
  cv2.line(image, (170,5), (217,5), (255,0,0), 2) 
  cv2.line(image, (25,5), (68,5), (255,0,0), 2)
  cv2.line(image, (99,5), (139,5), (255,0,0), 2)
  cv2.line(image, (170,5), (217,5), (255,0,0), 2)
  cv2.line(image, (422,5), (471,5), (255,0,0), 2)
  cv2.line(image, (479,5), (525,5), (255,0,0), 2)
  cv2.line(image, (531,5), (581,5), (255,0,0), 2)
  coordinates = [(45,5), (92,5),(99,5), (149,5),(295,5), (358,5),(422,5), (471,5),(479,5), (525,5),(531,5), (581,5)]
  coordinates_1 = [(45,5), (92,5),(99,5), (149,5),(295,5), (358,5),(422,5), (471,5),(479,5), (525,5),(531,5), (581,5) , (25,5), (68,5),(99,5), (139,5),(170,5), (217,5)]
  #coordinates_1 = [(25,5), (68,5),(99,5), (139,5),(170,5), (217,5)]
  edged = cv2.Canny(gray, 50, 100)
  edged = cv2.dilate(edged, None, iterations=1)
  edged = cv2.erode(edged, None, iterations=1)
  cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] 
  (cnts, _) = contours.sort_contours(cnts)
  colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0), (255, 0, 255))
  refObj = None
  for c in cnts:
      box = cv2.minAreaRect(c)
      box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
      box = np.array(box, dtype="int")
      box = perspective.order_points(box)
      cX = np.average(box[:, 0])
      cY = np.average(box[:, 1])
      if refObj is None:
          (tl, tr, br, bl) = box
          (tlblX, tlblY) = midpoint(tl, bl)
          (trbrX, trbrY) = midpoint(tr, br)
          D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
          refObj = (box, (cX, cY), D / 0.9)
          continue
      orig = image.copy()
      cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
      cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)
      

      refCoords = np.vstack([refObj[0], refObj[1]])
      objCoords = np.vstack([box, (cX, cY)])
      for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
          cv2.circle(orig, (int(xA), int(yA)), 5, color, -1)
          cv2.circle(orig, (int(xB), int(yB)), 5, color, -1)
          cv2.line(orig, (int(xA), int(yA)), (int(xB), int(yB)), color, 2)

          D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]
          (mX, mY) = midpoint((xA, yA), (xB, yB))
          cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
          cv2.waitKey(0)
          cv2.destroyAllWindows()
  def start():
      start = [(element) for element in _]

      return start
  def end():
      end = [str(element) for element in _]

      return end

  def coordinate(cnts):
      return [str(coord) for coord in cnts]
  array_of_text = coordinate(cnts)
  x = array_of_text
  c = []
  c.append(x)
  plain_list = [[list(arr) for arr in sub_list] for sub_list in cnts_list]
  # count the occurrences of each 2nd element
  counts = Counter([sublist[1][1] for sublist in plain_list])

  # get the two most common 2nd elements
  most_common = counts.most_common(3)
  common_second_elems = [elem[0] for elem in most_common]
  # filter the nested list to get only the sublists with the most common 2nd elements
  result = [sublist for sublist in plain_list if sublist[1][1] in common_second_elems]

  for i in range(len(result)):
      for j in range(len(result[i])):
          result[i][j][0] += 18
  if(temVar=='vert'):
    print(temVar)
    for i in range(len(result)):
        for j in range(len(result[i])):
            # Swap the elements in the current nested cell
            result[i][j][0] += 11
            result[i][j][0], result[i][j][1] = result[i][j][1], result[i][j][0]
    # Print the updated nested list
    for i in range(len(result)):
      for j in range(len(result[i])):
        result[i][j][0] = startcord
    print(result)
  else:
    for i in range(len(result)):
      for j in range(len(result[i])):
        result[i][j][1] = endcord
    print(result)

      
  # Pair corresponding elements
  paired_lst = [[result[i][0], result[i+1][0]] for i in range(0, len(result)-1, 2)]
  if len(result) % 2 != 0:
      paired_lst.append([result[-1][0]])
  int_list_result = [[[int(num) for num in inner_sublist] for inner_sublist in sub_list] for sub_list in paired_lst]
  return int_list_result
