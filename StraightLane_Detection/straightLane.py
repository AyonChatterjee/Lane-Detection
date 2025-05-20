import cv2 
import numpy as np
import math
import copy

def preprocessing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gblur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gblur, 150, 255, cv2.THRESH_BINARY)[1]
    return thresh

def regionOfInterest(img, polygon):
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], (255, 255, 255))
    else:
        cv2.fillPoly(mask, [np.array(polygon, dtype=np.int32)], 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def slopeIntercept(line):
    m = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
    b = line[1][1] - m * line[1][0]
    return m, b

def removeCloseLines(linelist, m):
    linelist_copy = copy.deepcopy(linelist)
    for line in linelist:
        try:
            m1, _ = slopeIntercept(line)
            if abs(m - m1) <= 0.5:
                linelist_copy.remove(line)
        except ZeroDivisionError:
            continue
    return linelist_copy

def lineDetection(img, masked_img, solid_line_previous, dashed_line_previous):
    img_copy = copy.deepcopy(img)
    height, width = masked_img.shape
    linesP = cv2.HoughLinesP(masked_img, 1, np.pi/180, 50, None, 30, 20)

    if linesP is None:
        return img_copy, solid_line_previous, dashed_line_previous

    linelist = linesP.tolist()
    linelist = [tuple((line[0][:2], line[0][2:])) for line in linelist]

    line_length = [math.dist(line[0], line[1]) for line in linelist]
    try:
        solid_line = linelist[line_length.index(max(line_length))]
        linelist.remove(solid_line)
    except ValueError:
        solid_line = solid_line_previous

    if solid_line is not None:
        try:
            m, b = slopeIntercept(solid_line)
            linelist = removeCloseLines(linelist, m)
            initial = (int((height * 0.6 - b) / m), int(height * 0.6))
            final = (int((height - b) / m), height)
            detected_line = cv2.line(img_copy, initial, final, (0, 255, 0), 5)
        except ZeroDivisionError:
            detected_line = img_copy
    else:
        detected_line = img_copy

    line_length = [math.dist(line[0], line[1]) for line in linelist]
    try:
        dashed_line = linelist[line_length.index(max(line_length))]
    except ValueError:
        dashed_line = dashed_line_previous

    if dashed_line is not None:
        try:
            m, b = slopeIntercept(dashed_line)
            initial = (int((height * 0.6 - b) / m), int(height * 0.6))
            final = (int((height - b) / m), height)
            detected_line = cv2.line(detected_line, initial, final, (0, 0, 255), 5)
        except ZeroDivisionError:
            pass

    return detected_line, solid_line, dashed_line

# ---- Main Program ----
video_path = r"C:\Users\chatt\Desktop\LANE Detection\StraightLane_Detection\Straight_Lane1.mp4"
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Error: Couldn't open video file.")
    exit()

frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('output_lane_detection.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

solid_line_previous = None
dashed_line_previous = None

print("Generating video output...\n")

while True:
    isTrue, img = video.read()
    if not isTrue:
        break

    processed_img = preprocessing(img)
    height, width = processed_img.shape
    polygon = [
        (int(width * 0.1), height),
        (int(width * 0.45), int(height * 0.6)),
        (int(width * 0.55), int(height * 0.6)),
        (int(0.95 * width), height)
    ]
    masked_img = regionOfInterest(processed_img, polygon)
    detected_lines, solid_line, dashed_line = lineDetection(img, masked_img, solid_line_previous, dashed_line_previous)
    solid_line_previous = solid_line
    dashed_line_previous = dashed_line
    out.write(detected_lines)

out.release()
video.release()
print("Video output generated successfully.\n")
