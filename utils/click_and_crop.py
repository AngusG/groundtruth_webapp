import cv2
import numpy as np
import argparse

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
w=0

crop_list=[]

# mouse callback function
def crop_regions(event,x,y,flags,param):
    global w

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.rectangle(img,(x-w,y-w),(x+w,y+w),(0,255,0),3)
        crop_list.append((x,y))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', help="path to image to segment")
    parser.add_argument('img_name', help="name of image to segment, JPG assumed")
    parser.add_argument('out_path', help="path to save segmented image")
    parser.add_argument('--wnd', type=int, help="crop width", default=100)
    parser.add_argument('--ds', type=int, help="image downsampling ratio", default=1)
    args = parser.parse_args()

    w=args.wnd

    input_file = args.img_path + args.img_name + ".JPG"
    img = cv2.imread(input_file)[::args.ds,::args.ds,:].astype(np.uint8)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',crop_regions)

    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == ord('q'):
            break

    i=0
    for x,y in crop_list:
        roi = img[y-w:y+w,x-w:x+w,:]
        out_file = args.out_path+args.img_name+'_w'+str(w) \
                +'_ds'+str(args.ds)+'_'+str(i)+"_.JPG"
        cv2.imwrite(out_file,roi)
        i+=1

    cv2.destroyAllWindows()