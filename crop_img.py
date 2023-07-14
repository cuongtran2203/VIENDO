import cv2
import numpy as np
from detector.YOLO_detector import Detector
import os
class PreProcess():
    def __init__(self,model_path=None) :
        self.detect=Detector(model_path)
    def crop(self,img):
        bbox=self.detect.detect(img)
        if len(bbox)>0 :
            for box in bbox :
                box=list(map(int,box))
                crop_img=img[box[1]:box[3],box[0]:box[2]]
                h,w=crop_img.shape[:2]
                if h/w>1:
                    return crop_img[0:int(h/2),0:w]
                else:
                    img_part1=crop_img[0:int(h/2),0:int(w/2)]
                    img_part2=crop_img[0:int(h/2),int(w/2):w]
                    return img_part1, img_part2
        return None
                # image_crop_path=os.path.join("crop_image",im_file)
                # cv2.imwrite(image_crop_path,crop_img)
                # print("saved")
if __name__=="__main__":
    # detect=Detector("./weights/best.onnx")
    # dir="images"
    # img_list=os.listdir(dir)
    # for im_file in img_list :
    #     im_path=os.path.join(dir,im_file)
    #     img=cv2.imread(im_path)
    #     bbox=detect.detect(img)
    #     if bbox is not None :
    #         for box in bbox :
    #             box=list(map(int,box))
    #             crop_img=img[box[1]:box[3],box[0]:box[2]]
    #             image_crop_path=os.path.join("crop_image",im_file)
    #             cv2.imwrite(image_crop_path,crop_img)
    #             print("saved")
    crop=PreProcess(model_path="./weights/best.onnx")
    dir="images"
    dir2="crop_haft"
    img_list=os.listdir(dir)
    for im_file in img_list :
        im_path=os.path.join(dir,im_file)
        img=cv2.imread(im_path)
        if img is not None :
            img_part=crop.crop(img)
            if img_part is not None :
                if len(img_part)==2:
                    img_part1_path=os.path.join(dir2,im_file.replace(".jpg","_1.jpg"))
                    img_part2_path=os.path.join(dir2,im_file.replace(".jpg","_2.jpg"))
                    cv2.imwrite(img_part1_path,img_part[0])
                    cv2.imwrite(img_part2_path,img_part[1])
                else:
                    img_part1_path=os.path.join(dir2,im_file)
                    cv2.imwrite(img_part1_path,img_part)
                
                print("saved")
            
