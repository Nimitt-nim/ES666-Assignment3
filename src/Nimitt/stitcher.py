import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import imutils

class PanaromaStitcher():
    def __init__(self):
        pass

    def make_panaroma_for_images_in(self, path):
        # Get all images from the folder path
        all_images = sorted(glob.glob(path + os.sep + '*'))
        print(f'Found {len(all_images)} images for stitching')

        # Initialize an empty list to store homography matrices
        homography_matrix_list = []
           
        stitched_image = None

        for i in range(len(all_images)-1,0,-1 ):

            image = cv2.imread(all_images[i])
            image = imutils.resize(image, width=400)
            
            if stitched_image is None:
                stitched_image = image  
                continue
            else:
                stitched_image,H = self.stitch([stitched_image, image])
            homography_matrix_list.append(H)

        return stitched_image, homography_matrix_list

    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
        ### Getting SIFT Features and descriptors
        (imageA, imageB) = images
        (kpsA, featuresA) = self.sift_detector(imageA)
        (kpsB, featuresB) = self.sift_detector(imageB)
        ### Computing H using RANSAC and detected SIFT
        H = self.match_keypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        result = self.warp_perspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[:, 0:400] = imageB

        return result,H
    
    def warp_perspective(self,image, H, dsize):
        # Applying H_inv to image to map to orientation of first image 
        w, h = dsize
        result = np.zeros((h, w, 3), dtype=image.dtype)

        H_inv = np.linalg.inv(H)

        for y in range(h):
            for x in range(w):
                pt = np.array([x, y, 1.0])
                src_pt = H_inv @ pt
                src_pt /= src_pt[2]

                src_x, src_y = int(src_pt[0]), int(src_pt[1])
                if 0 <= src_x < image.shape[1] and 0 <= src_y < image.shape[0]:
                    result[y, x] = image[src_y, src_x]
            
        return result

    def sift_detector(self, image):
        # Getting SFIT features 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        descriptor = cv2.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(gray, None)
        
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)
    
    def normalize_points(self,points):
        # Normalizing 
        centroid = np.mean(points, axis=0)
        avg_dist = np.mean(np.linalg.norm(points - centroid, axis=1))
        scale = np.sqrt(2) / avg_dist
        T = np.array([
            [scale, 0, -scale * centroid[0]],
            [0, scale, -scale * centroid[1]],
            [0, 0, 1]
        ])
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        normalized_points = (T @ points_homo.T).T[:, :2]
        return normalized_points, T


    def find_homography(self,src_pts, dst_pts, reproj_thresh=4.0, max_iters=2000, confidence=0.99):
        # Computing H using DIT with RANSAC

        src_pts = np.array(src_pts)
        dst_pts = np.array(dst_pts)
        num_pts = src_pts.shape[0]
        
        best_H = None
        max_inliers = 0
        src_pts_norm, T_src = self.normalize_points(src_pts)
        dst_pts_norm, T_dst = self.normalize_points(dst_pts)

        # RANSAC loop
        for _ in range(max_iters):
            idx = np.random.choice(num_pts, 4, replace=False)
            src_sample = src_pts_norm[idx]
            dst_sample = dst_pts_norm[idx]

            A = []
            for (src, dst) in zip(src_sample, dst_sample):
                x, y = src
                u, v = dst
                A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
                A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
            A = np.array(A)
            _, _, V = np.linalg.svd(A)
            H_norm = V[-1].reshape((3, 3))


            H = np.linalg.inv(T_dst) @ H_norm @ T_src

            src_pts_homo = np.hstack([src_pts, np.ones((num_pts, 1))])
            projected_pts = (H @ src_pts_homo.T).T
            epsilon = 1e-8  

            projected_pts = projected_pts[:, :2] / (projected_pts[:, 2:3] + epsilon)
            errors = np.linalg.norm(projected_pts - dst_pts, axis=1)
            inliers = errors < reproj_thresh
            num_inliers = np.sum(inliers)

            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_H = H
            if max_inliers / num_pts >= confidence:
                break

        return best_H, inliers



    def match_keypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            (H, status) =self.find_homography(ptsA, ptsB)
            return  H
    


# p = PanaromaStitcher()
# image, hms = p.make_panaroma_for_images_in('/Users/nimitt/Documents/ES666-Assignment3/Images/I1')

# plt.imshow(image)
# plt.show()