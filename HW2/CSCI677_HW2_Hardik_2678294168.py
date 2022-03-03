import cv2
import os
import argparse
import numpy as np
import xml.etree.ElementTree as ET


def selective_search(img, strategy):
    """
    @brief Selective search with different strategies
    @param img The input image
    @param strategy The strategy selected ['color', 'all']
    @retval bboxes Bounding boxes
    """
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    gs = cv2.ximgproc.segmentation.createGraphSegmentation()
    ##################################################
    # TODO: For this part, please set the K as 200,  #
    #       sigma as 0.8 for the graph segmentation. #
    #       Use gs as the graph segmentation for ss  #
    #       to process after strategies are set.     #
    ##################################################

    #converting image from bgr to rgb format
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    #feeding image
    ss.addImage(img_rgb)

    #setting paramater values, K & sigma for graphSegmentation
    gs.setK(200)
    gs.setSigma(0.8)

    #adding graph segmentation to selectiveSearchSegmentation
    ss.addGraphSegmentation(gs)

    #checking the input for which strategy to be implemented
    if strategy=='color':
        color_strategy=cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
        ss.addStrategy(color_strategy)
    elif strategy=='all':
        color_strategy=cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
        size_strategy=cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
        texture_strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
        fill_strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
        multiple_strategy = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(color_strategy,size_strategy,texture_strategy,fill_strategy)
        ss.addStrategy(multiple_strategy)

    ##################################################
    # End of TODO                                    #
    ##################################################
    bboxes = ss.process()
    xyxy_bboxes = []

    for box in bboxes:
        x, y, w, h = box
        xyxy_bboxes.append([x, y, x + w, y + h])

    return xyxy_bboxes


def parse_annotation(anno_path):
    """
    @brief Parse annotation files for ground truth bounding boxes
    @param anno_path Path to the file
    """
    tree = ET.parse(anno_path)
    root = tree.getroot()
    gt_bboxes = []
    for child in root:
        if child.tag == 'object':
            for grandchild in child:
                if grandchild.tag == "bndbox":
                    x0 = int(grandchild.find('xmin').text)
                    x1 = int(grandchild.find('xmax').text)
                    y0 = int(grandchild.find('ymin').text)
                    y1 = int(grandchild.find('ymax').text)
                    gt_bboxes.append([x0, y0, x1, y1])
    return gt_bboxes


def bb_intersection_over_union(boxA, boxB):
    """
    @brief compute the intersaction over union (IoU) of two given bounding boxes
    @param boxA numpy array (x_min, y_min, x_max, y_max)
    @param boxB numpy array (x_min, y_min, x_max, y_max)
    """
    ##################################################
    # TODO: Implement the IoU function               #
    ##################################################
    #calculating the top-left and bottom-right coordinates of intersecting rectangle
    x1_intersect=max(boxA[0],boxB[0])
    y1_intersect=max(boxA[1],boxB[1])
    x2_intersect=min(boxA[2],boxB[2])
    y2_intersect=min(boxA[3],boxB[3])

    #calculating Area of intersecting rectangle
    numerator=max(0,x2_intersect-x1_intersect+1)*max(0,y2_intersect-y1_intersect+1)

    #Calculating the areas of Union of gt and predicted rectangles
    boxA_area=(boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
    boxB_area=(boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
    denominator=float(boxA_area+boxB_area-numerator)

    #Calculating IoU
    iou=numerator/denominator

    ##################################################
    # End of TODO                                    #
    ##################################################
    return iou


def visualize(img, boxes, color):
    """
    @breif Visualize boxes
    @param img The target image
    @param boxes The box list
    @param color The color
    """
    for box in boxes:
    ##################################################
    # TODO: plot the rectangles with given color in  #
    #       the img for each box.                    #
    ##################################################
        img=cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),color)

    ##################################################
    # End of TODO                                    #
    ##################################################
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='color')

    args = parser.parse_args()
    img_dir = './HW2_Data/JPEGImages'
    anno_dir = './HW2_Data/Annotations'
    thres = .5

    img_list = os.listdir(img_dir)
    num_hit = 0
    num_gt = 0

    for img_path in img_list:
        """
        Load the image file here through cv2.imread
        """
        img_id = img_path[:-4]
        img_name = os.path.join(img_dir, img_path)
        ##################################################
        # TODO: Load the image with OpenCV               #
        ##################################################
        img = cv2.imread(img_name)

        ##################################################
        # End of TODO                                    #
        ##################################################

        proposals = selective_search(img, args.strategy)
        gt_bboxes = parse_annotation(os.path.join(anno_dir, img_id + ".xml"))
        iou_bboxes = []  # proposals with IoU greater than 0.5
        #print("length of proposal box ",len(proposals))
        ##################################################
        # TODO: For all the gt_bboxes in each image,     #
        #       please calculate the recall of the       #
        #       gt_bboxes according to the document.     #
        #       Store the bboxes with IoU >= 0.5         #
        #       If there is more than one proposal has   #
        #       IoU >= 0.5 with a same groundtruth bbox, #
        #       store the one with biggest IoU.          #
        ##################################################
        for gt in gt_bboxes:
            best_iou=0
            best_proposal_box=0
            for proposal_box in proposals:
                #calculating IoU score for each proposal and GT box
                iou_score=bb_intersection_over_union(gt, proposal_box)
                if iou_score>=0.5:
                    #checking if present IoU score is better than last stored proposal box
                    if iou_score>best_iou:
                        best_iou=iou_score
                        best_proposal_box=proposal_box
            if best_iou!=0:
                #storing the best proposal box i.e with the highest IoU>0.5 for each GT box
                iou_bboxes.append(best_proposal_box)
        #print('iou-bboxes',len(iou_bboxes))
        #print('gt_bboxes',len(gt_bboxes))

        #calculating recall score
        recall=len(iou_bboxes)/len(gt_bboxes)
        ##################################################
        # End of TODO                                    #
        ##################################################

        vis_img = img.copy()
        vis_img = visualize(vis_img, gt_bboxes, (255, 0, 0))
        vis_img = visualize(vis_img, iou_bboxes, (0, 0, 255))

        proposals_img = img.copy()
        proposals_img = visualize(proposals_img, gt_bboxes, (255, 0, 0))
        proposals_img = visualize(proposals_img, proposals, (0, 0, 255))

        ##################################################
        # TODO: (optional) You may use cv2 to visualize  #
        #       or save the image for report.            #
        ##################################################

        #adding text on output image, mentioning the recall score
        vis_img=cv2.putText(vis_img, 'Recall: {}'.format(recall), (18,15), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow("Proposals_Image",proposals_img)
        cv2.imshow("Image",vis_img)
        cv2.waitKey(0)

        #filepath for saving the output images for visualization and report
        file_save_proposals="./HW2_Data/Results/Proposals/ " +args.strategy + img_path +".jpg"
        file_save_image = "./HW2_Data/Results/Images/ " +args.strategy + img_path + ".jpg"

        #Saving the images to the defined path
        cv2.imwrite(file_save_proposals,proposals_img)
        cv2.imwrite(file_save_image,vis_img)
        ##################################################
        # End of TODO                                    #
        ##################################################


if __name__ == "__main__":
    main()




