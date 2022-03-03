#Importing required packages
import cv2
import numpy as np

file_idx=0
def resize_img(filename,scale):
    img=cv2.imread(filename)
    scale_percent = scale
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def visualize(img):
    cv2.imshow('window',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(img_src,img_dst,filedest):
    global file_idx
    file_path_src=filedest+'_'+str(file_idx)+'.jpg'
    cv2.imwrite(file_path_src,img_src)
    file_idx += 1
    file_path_dst = filedest + '_' + str(file_idx) + '.jpg'
    cv2.imwrite(file_path_dst, img_dst)
    file_idx += 1

def experiment(src,dst,kmatch,image_name):
    #defininng filepath for saving images
    path_src = './HW3_Results/'+image_name

    #convert images to grayscale
    src_img=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    dst_img = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    #creating SIFT
    sift=cv2.xfeatures2d.SIFT_create()
    keypoints_src,descriptor_src=sift.detectAndCompute(src_img,None)
    keypoints_dst,descriptor_dst=sift.detectAndCompute(dst_img,None)

    print("# SIFT keypoints in source image = {}".format(len(keypoints_src)))
    print("# SIFT keypoints in destination image = {}".format(len(keypoints_dst)))

    #drawing features on images
    out_src=cv2.drawKeypoints(src_img,keypoints_src,outImage=src_img)
    out_dst=cv2.drawKeypoints(dst_img,keypoints_dst,outImage=dst_img)

    #visualize images
    visualize(out_src)
    visualize(out_dst)

    #save images
    path=path_src+'SIFT'
    save_image(out_src,out_dst,path)

    #feature matching- RANSAC
    bf=cv2.BFMatcher(cv2.NORM_L2,crossCheck=False)
    matches=bf.knnMatch(descriptor_src,descriptor_dst,k=kmatch)

    #applying ratio test
    good_matches=[]
    for i,j in matches:
        if i.distance<0.75*j.distance:
            good_matches.append(i)

    good_matches=sorted(good_matches,key=lambda x:x.distance)
    print("# of total matches found = {}".format(len(matches)))
    print("# of good matches found = {}".format(len(good_matches)))

    # draw top20 matches
    out_matches=cv2.drawMatches(src_img,keypoints_src,dst_img,keypoints_dst,good_matches[:20],None,flags=2)
    visualize(out_matches)

    # save images
    path = path_src + 'top20'
    save_image(out_matches, out_dst, path)

    #finding homography matrix
    if len(good_matches)>10:
        src_pts = np.float32([keypoints_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_dst[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchMask = mask.ravel().tolist()
        h, w = src_img.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    else:
        print("Not enough matches are found")
        matchMask=None
    print("The homography matrix is {}".format(M))
    print("# of inliner matches after Homography estimation : {}".format(np.sum(matchMask)))
    dst = cv2.perspectiveTransform(pts,M)
    dst_img = cv2.polylines(dst_img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    visualize(dst_img)

    matches_masked=[]
    for i, match in enumerate(good_matches):
        if matchMask[i]==1:
            matches_masked.append(match)
    print("# of matches found after Homography={}".format(len(matches_masked)))
    matches_masked=sorted(matches_masked,key=lambda x:x.distance)
    out_masked_matches=cv2.drawMatches(src_img,keypoints_src,dst_img,keypoints_dst,matches_masked[:10],None,flags=2)

    #draw top10 matches that have minimum error between projected source keypoint and destination keypoint
    visualize(out_masked_matches)

    #save images
    path = path_src + 'top10'
    save_image(out_masked_matches, out_dst, path)

    #drawing all inliners
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchMask,  # draw only inliers
                       flags=2)
    out_inliners = cv2.drawMatches(src_img, keypoints_src, dst_img, keypoints_dst, good_matches,None,**draw_params)

    #draw all the inliners
    visualize(out_inliners)

    #save the image showing all the inliners
    path = path_src + 'inliners'
    save_image(out_inliners, out_dst, path)


def main():
    #specifying filename of source and target images
    src0 = './HW3_Data/src_0.jpg'
    src1 = './HW3_Data/src_1.jpg'
    src2 = './HW3_Data/src_2.jpg'
    dst0 = './HW3_Data/dst_0.jpg'
    dst1 = './HW3_Data/dst_1.jpg'

    #reading the resized image
    s0=resize_img(src0,100)
    s1=resize_img(src1,100)
    s2=resize_img(src2,100)
    d0=resize_img(dst0,100)
    d1=resize_img(dst1,100)

    experiment(s0,d0,2,'s0d0')
    experiment(s0,d1,2,'s0d1')
    experiment(s1,d0,2,'s1d0')
    experiment(s1,d1,2,'s1d1')
    experiment(s2,d0,2,'s2d0')
    experiment(s2,d1,2,'s2d1')

if __name__ == "__main__":
    main()


