import csv
import glob
from pathlib import Path
import numpy as np
from string import digits
from skimage.feature import local_binary_pattern
from scipy.spatial import ConvexHull
import cv2

#init default values

#example of accepted values, set it in start.py
#DATASET_PATH = '../dataset/raw/*.jpg'
#OUTPUT_FILE = "all_features.csv"
#DATASET_PATH = '../dataset/testing/gambar_ttrpld.jpg'

COLOR_RED   = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE  = (255, 0, 0)

def extract(DATASET_PATH, OUTPUT_FILE):  
    
    #init data container
    data        = []

    #init LBP
    METHOD      = 'default'
    radius      = 1
    n_points    = 8 * radius

    #begin
    for filename in glob.glob(DATASET_PATH):
        print("===========================")
        #read image as BGR
        img = cv2.imread(filename)

        #get the filename
        true_filename = Path(filename).name

        #create uniform label with capitalized word and without underscore(s) and number(s)
        label = Path(filename).stem
        remove_digits = str.maketrans('', '', digits)
        label = label.replace('_', '').translate(remove_digits).lower().capitalize()

        print("Processing " + str(true_filename))
        #general preprocessing
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
        ret, thresh = cv2.threshold(blurred_img, 180, 255, cv2.THRESH_BINARY)
        thresh_inverted = (255 - thresh)

        #======================PROCESSING======================
        print("Processing morphological features...")
        # Morphology - find contour
        contours, hierarchy = cv2.findContours(thresh_inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_with_contour = cv2.drawContours(img, contours, -1, COLOR_RED, 2)
        contour = contours[0]

        # Morphology - fit to bounding rectangle
        x, y, length, width = cv2.boundingRect(contour)
        starting_point = (x,y)
        ending_point = (x+length, y+width)
        cv2.rectangle(img_with_contour, starting_point, ending_point, COLOR_BLUE, 1)
        #swap values if width value is larger than length
        if width > length:
            length, width = width, length

        # Morphology - fit to bounding ellipse
        #ellipse = cv2.fitEllipse(contour)
        #cv2.ellipse(img_with_contour, ellipse, COLOR_GREEN, 1)
        
        # Morphology - get raw data
        leaf_perimeter = cv2.arcLength(contour, True)
        leaf_area = cv2.contourArea(contour)
        leaf_diameter = np.sqrt(4 * leaf_area / np.pi)
        #source: http://158.110.32.35/CLASS/IMP-CHIM/PGSF21-42.pdf
        #CHAPTER  1, Size and Properties of Particles
        #projected area diameter, page 6, (1,1)
        leaf_length = length
        leaf_width = width

        print("Processing LBP features...")
        # LBP
        lbp = local_binary_pattern(gray_img, n_points, radius, METHOD)
        (hist, _) = np.histogram(lbp.ravel(),
                                     density=True,
                                     bins=26,
                                     range=(0, 10))

        # LBP - normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum()+ 1e-7)

        print("Processing convex hull features...")
        # Convex Hull - generate edges
        edges = cv2.Canny(img, 100, 200)
        points = []
        for y in range(0, edges.shape[0]):
            for x in range(0, edges.shape[1]):
                if edges[y, x] != 0:
                    points = points + [[x, y]]
        points = np.array(points)
        hull = ConvexHull(points)

        # Convex Hull - get raw data
        convex_perimeter = hull.area
        convex_area = hull.volume
        #for 2D data, ConvexHull.area returns perimeter, and ConvexHull.volume returns area
        #source: https://stackoverflow.com/questions/35664675/in-scipys-convexhull-what-does-area-measure
        #archive.org link: https://web.archive.org/web/20160611124656/https://stackoverflow.com/questions/35664675/in-scipys-convexhull-what-does-area-measure
        #quote: "Volume and area are 3d concepts, but your data is 2d - a 2x2 square. Its area is 4, and perimeter is 8 (the 2d counterparts)."      

        #======================FEATURES======================
        print("Calculating all features...")

        # Morphology - features
        aspect_ratio                            = float(leaf_length / leaf_width)
        form_factor                             = ((4 * np.pi) * leaf_area) / (leaf_perimeter**2)
        rectangularity                          = (leaf_length * leaf_width) / leaf_area
        narrow_factor                           = leaf_diameter / leaf_length
        perimeter_ratio_of_diameter             = leaf_perimeter / leaf_diameter
        perimeter_ratio_of_length_and_width     = leaf_perimeter / (leaf_length + leaf_width)
        
        # LBP - features
        # features are hist[0] ... hist[25]

        # Convex Hull - features
        convexity                               = convex_perimeter / leaf_perimeter
        solidity                                = leaf_area / convex_area

        #======================EXPORTING======================
        print("Appending all features to main data list...")
        featurelist = []
        featurelist.append(true_filename)
        featurelist.append(label)
        
        # LBP
        for value in hist:
            featurelist.append(value)

        # Morphology
        featurelist.append(aspect_ratio)
        featurelist.append(form_factor)
        featurelist.append(rectangularity)
        featurelist.append(narrow_factor)
        featurelist.append(perimeter_ratio_of_diameter)
        featurelist.append(perimeter_ratio_of_length_and_width)

        # Convex Hull
        featurelist.append(convexity)
        featurelist.append(solidity)

        data.append(featurelist)
        #======================DEBUGGING======================

        #cv2.imshow("img", img_with_contour)
        #cv2.waitKey(0)
    print("Saving result to csv...")
    save(data, OUTPUT_FILE)

def save(data_list, OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for row in data_list:
                writer.writerow(row)
        print("Saving success!")
    except BaseException as e:
        print("Saving failed! "+str(e))

if __name__ == "__main__":
    DATASET_PATH = '../dataset/testing/*.jpg'
    #DATASET_PATH = '../dataset/testing/8b02963f832e43549acf90b90fe010e2.jpg'
    OUTPUT_FILE = "../dataset/csv/DEBUG_training_features.csv"
    extract(DATASET_PATH, OUTPUT_FILE)