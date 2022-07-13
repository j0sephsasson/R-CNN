import os, cv2, matplotlib.pyplot as plt, numpy as np, pandas as pd
from tqdm import tqdm

class RCNN:
  def __init__(self, image_dir, label_dir):
    self.image_dir = image_dir
    self.label_dir = label_dir
    self.ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

  def get_iou(self, bb_gt, bb_pred):
    """
    This function computes the iou (intersection over union)

    PARAMS:
    bb_gt --> type: dict, contains labeled bounding boxes
    bb_pred --> type: dict, contains predicted bounding boxes

    RETURNS:
    iou --> type: int, the iou score (accuracy)
    """

    assert bb_gt['x1'] < bb_gt['x2']
    assert bb_gt['y1'] < bb_gt['y2']
    assert bb_pred['x1'] < bb_pred['x2']
    assert bb_pred['y1'] < bb_pred['y2']

    x_left = max(bb_gt['x1'], bb_pred['x1'])
    y_top = max(bb_gt['y1'], bb_pred['y1'])
    x_right = min(bb_gt['x2'], bb_pred['x2'])
    y_bottom = min(bb_gt['y2'], bb_pred['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb_gt_area = (bb_gt['x2'] - bb_gt['x1']) * (bb_gt['y2'] - bb_gt['y1'])
    bb_pred_area = (bb_pred['x2'] - bb_pred['x1']) * (bb_pred['y2'] - bb_pred['y1'])

    iou = intersection_area / float(bb_gt_area + bb_pred_area - intersection_area)

    assert iou >= 0.0
    assert iou <= 1.0
    return iou

  def create_training_set(self):
    assert(len(os.listdir(self.image_dir)) == len(os.listdir(self.label_dir))) ## make sure all images have a label

    X_train = []
    y_train = []
    
    for idx, file_ in enumerate(tqdm(os.listdir(self.label_dir), desc='PROCESSING IMAGES', unit='IMAGE')):
        if file_.startswith("airplane"):
            filename = file_.split(".")[0]+".jpg"
            image = cv2.imread(os.path.join(self.image_dir, filename)) ## read in image
            df = pd.read_csv(os.path.join(self.label_dir, file_)) ## read in labels

            gt_values=[] ## ground_truth bounding box labels
            for row in df.iterrows():
                x1 = int(row[1][0].split(" ")[0])
                y1 = int(row[1][0].split(" ")[1])
                x2 = int(row[1][0].split(" ")[2])
                y2 = int(row[1][0].split(" ")[3])
                gt_values.append({"x1":x1,"x2":x2,"y1":y1,"y2":y2})

            self.ss.setBaseImage(image)
            self.ss.switchToSelectiveSearchFast()
            ssresults = self.ss.process()
            imout = image.copy()

            counter = 0 ## collect at a maximum 30 positive samples from a single image
            falsecounter = 0 ## collect at a maximum 30 negative samples from a single image

            for idx, prediction in enumerate(ssresults): ## loop over all recognized objects in the image
                if idx < 2000:
                    for gtval in gt_values: ## loop over all ground_truth labels in this image
                        x,y,w,h = prediction
                        iou = self.get_iou(gtval, {"x1":x,"x2":x+w,"y1":y,"y2":y+h}) ## calculate our iou
                        if counter < 30: ## collect only 30 samples negative or positive
                            if iou > 0.70: ## if iou is above 70% --> we have identified an airplane
                                timage = imout[y:y+h,x:x+w] ## create a new image with only the specified coordinates of our object (airplane)
                                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                X_train.append(resized)
                                y_train.append(1)
                                counter += 1
                        if falsecounter < 30:
                            if iou < 0.3: ## if iou is less than 30% --> we have not identified an airplane
                                timage = imout[y:y+h,x:x+w]
                                resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
                                X_train.append(resized)
                                y_train.append(0)
                                falsecounter += 1

    return np.array(X_train), np.array(y_train)

  def single_image_ss(self, img_path):
     im = cv2.imread(img_path)

     self.ss.setBaseImage(im)
     self.ss.switchToSelectiveSearchFast()
     rects = self.ss.process()
     imOut = im.copy()

     for i, rect in (enumerate(rects)):
         x, y, w, h = rect
         cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

     plt.imshow(imOut)

  def predict(self, img_path, label_path):
    """
    Just showing how it will work --> this is not implemented 
    """
    
    df = pd.read_csv(label_path)
    img = cv2.imread(img_path) ## read in image

    for row in df.iterrows():
      x1 = int(row[1][0].split(" ")[0])
      y1 = int(row[1][0].split(" ")[1])
      x2 = int(row[1][0].split(" ")[2])
      y2 = int(row[1][0].split(" ")[3])
      cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0), 2)

    plt.figure()
    plt.imshow(img)