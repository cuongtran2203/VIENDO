import cv2
import numpy as np
def empty(a):
    print("")
img=cv2.imread("crop\\200612aa_tngx0003_tx7-rx714_.jpg")
img=cv2.resize(img,(1920,1080))
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",640,240)
# cv2.getTrackbarPos("HUE MIN","HSV",0,179,empty)
cv2.createTrackbar("HUE MIN","HSV",0,179,empty)
cv2.createTrackbar("HUE MAX","HSV",179,179,empty)
cv2.createTrackbar("SAT MIN","HSV",0,255,empty)
cv2.createTrackbar("SAT MAX","HSV",255,255,empty)
cv2.createTrackbar("VALUE MIN","HSV",0,255,empty)
cv2.createTrackbar("VALUE MAX","HSV",255,255,empty)
while True :
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min=cv2.getTrackbarPos("HUE MIN","HSV")
    h_max=cv2.getTrackbarPos("HUE MAX","HSV")
    s_min=cv2.getTrackbarPos("SAT MIN","HSV")
    s_max=cv2.getTrackbarPos("SAT MAX","HSV")
    v_min=cv2.getTrackbarPos("VALUE MIN","HSV")
    v_max=cv2.getTrackbarPos("VALUE MAX","HSV")
    lower=np.array([h_min,s_min,v_min])
    upper=np.array([h_max,s_max,v_max])
    mask=cv2.inRange(img_hsv,lower,upper)
    result=cv2.bitwise_and(img,img,mask=mask)
    cv2.imshow("Hsv",img_hsv)
    cv2.imshow("mask",result)
    if cv2.waitKey(1) & 0xFF==ord("q"):
        break