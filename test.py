import numpy as np
import cv2
import math

poly_interest =-1
#poly_interest = 43



def capture():
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        cv2.imshow('preview', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('sampleInput.png', frame)
            cv2.destroyAllWindows()
            break

# capture()


def calc_distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def calc_angle(a, b):
    return math.atan2(b[1]-a[1], b[0]-a[0])*180/math.pi


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def deviation(a, b, c):
    ab = calc_distance(a, b)
    bc = calc_distance(b, c)
    ac = calc_distance(a, c)

    # if obtuse triangle then point b doesnt lie between linesegment ac
    if(ab >= ac or bc >= ac):
        ret = False
        dist = 0
        return ret, dist

    ratio = ab/(ab+ac)
    ap = ac * ratio
    d = math.sqrt(ab**2 - ap**2)
    ret = True
    dist = d
    return ret, dist


def findLines(cont):
    validLineThreshold = 5
    distanceToleraneThreshold = 1

    i = 0
    len_c = len(cont)
    len_cm = int(len_c*1.3)

    predictedDistances = []
    predictedLines = []
    pointLineSegments = []
    pointLineSegment = []
    predictedLine = None

    acc_line = []
    lineWasFound = False

    pa = None
    pb = None
    pc = None

    dis_ab = None
    dis_bc = None
    dis_ac = None

    ang_ab = None
    ang_bc = None
    ang_ac = None

    isFirstLine = True

    while i < len_cm:
        if i+2 > len_c and len(predictedLines) > 0 and (i+2) % len_c > predictedLines[0][0] % len_c:
            break

        if type(pa) == type(None):
            pa = cont[i % len_c][0]
        if type(pb) == type(None):
            pb = cont[(i+1) % len_c][0]
        if type(pc) == type(None):
            pc = cont[(i+2) % len_c][0]

        if type(dis_ab) == type(None):
            dis_ab = calc_distance(pa, pb)
        if type(dis_bc) == type(None):
            dis_bc = calc_distance(pb, pc)
        if type(dis_ac) == type(None):
            dis_ac = calc_distance(pa, pc)

        #if type(ang_ab) == type(None):
        #   ang_ab = calc_angle(pa, pb)
        #if type(ang_bc) == type(None):
        #   ang_bc = calc_angle(pb, pc)
        #if type(ang_ac) == type(None):
        #   ang_ac = calc_angle(pa, pc)

        didMerge = False

        dis_err = dis_ab + dis_bc - dis_ac

        if dis_err <= distanceToleraneThreshold:
            didMerge = True

        if didMerge:
            if not isFirstLine:
                if not lineWasFound:
                    predictedLine = [(i) % len_c, (i+1) % len_c]
                    predictedLines.append(predictedLine)

                    pointLineSegment = [pa, pc]
                    pointLineSegments.append(pointLineSegment)

                    predictedDistances.append(dis_ac)

                else:
                    predictedLine[1] = (i+1) % len_c
                    pointLineSegment[1] = pc

                    predictedDistances[len(predictedDistances)-1] = dis_ac

            if len(acc_line) == 0:
                acc_line.append(i)
            acc_line.append(i+1)

            # change variable like
            #
            # A--B--C--d
            #     |
            #     V
            # A-----B==C

            # pa carry over
            pb = pc
            pc = None

            dis_ab = dis_ac
            dis_bc = None
            dis_ac = None

            lineWasFound = True

        else:
            # remove predicted line if its too short
            if type(predictedLine) != type(None):
                pl1 = cont[predictedLine[0] % len_c][0]
                pl2 = cont[(predictedLine[1]+1) % len_c][0]

                if calc_distance(pl1, pl2) < validLineThreshold:
                    predictedLines.pop()
                    predictedDistances.pop()
                    pointLineSegments.pop()
                predictedLine = None

            # no line was found by merger
            lineWasFound = False

            # check if the line B-C is long enough
            # to be considered an individual line itself
            if dis_bc >= validLineThreshold:
                if not isFirstLine:
                    predictedLine = [(i+1) % len_c, (i+1) % len_c]
                    predictedLines.append(predictedLine)

                    pointLineSegment = [pb, pc]
                    pointLineSegments.append(pointLineSegment)

                    predictedDistances.append(dis_bc)
                    lineWasFound = True

            # change variable like
            #
            # A--B--C--D
            #     |
            #     V
            # z--A--B==C

            pa = pb
            pb = pc
            pc = None

            dis_ab = dis_bc
            dis_bc = None
            dis_ac = None

            isFirstLine = False
            # end if
        i += 1
        # end while i < len_cc

    return predictedLines, pointLineSegments, predictedDistances


def checkIfQuadrilateral(lineIndeces, lines, distances):
    len_c = len(lines)
    if len_c < 4:
        return None, None

    i = 0
    copyDist = distances.copy()

    copyDist.sort(reverse=True)
    minDistance = copyDist[3]
    cutOutDistance = int(minDistance*0.20)

    linePairs = []

    while i < len_c:
        dis = distances[i]
        line = lines[i]

        if dis >= minDistance:
            linePairs.append(line)
        elif dis > cutOutDistance:
            return None, None
        i += 1

    len_p = len(linePairs)
    if len_p != 4:
        return None, None

    firstAngle = 0
    i = 0
    while i < len_p:
        pa = linePairs[i][0]
        pb = linePairs[i][1]
        pc = linePairs[(i+1) % len_p][0]
        pd = linePairs[(i+1) % len_p][1]

        ang1 = calc_angle(pa, pb)
        ang2 = calc_angle(pc, pd)

        del_ang = ang2 - ang1
        # if ang1 < -90 and ang2 > 90:
        #     del_ang -= 360
        # elif ang2 < -90 and ang1 > 90:
        #     del_ang += 360

        if del_ang>180:
            del_ang -= (math.floor(del_ang / 360) + 1) * 360

        elif del_ang<-180:
            del_ang += (math.floor(abs(del_ang) / 360) + 1) * 360

        if firstAngle == 0:
            if del_ang > 0:
                firstAngle = 1
            else:
                firstAngle = -1
        else:
            if firstAngle * del_ang <= 0:
                return None, None
        i += 1
    return linePairs, firstAngle


debugit = False
ind = 0
mind = -1
printit = False


def qrcodescan(frame):
    global ind, printit, mind, debugit
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    thr = math.floor(frame.mean()*2/3)
    ret, thresh = cv2.threshold(gray, thr, 255, 0)
    thresh = cv2.blur(thresh, (3, 3))
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if poly_interest != -1:
        contours = [contours[poly_interest]]

    if not debugit:
        for cont in contours:
            predictedLines, pointLineSegments, distances = findLines(cont)
            lines, direction = checkIfQuadrilateral(
                predictedLines, pointLineSegments, distances)

            if type(lines) == type(None):
                colors = [
                    (0, 0, 255),
                    (0x85, 0xbd, 0x03),
                    (0x4e, 0x36, 0x92),
                    (0x02, 0xdd, 0xfe),
                    (0xec, 0x85, 0x5c),
                    (0x46, 0x02, 0xfb)
                ]

                color_i = -1
                color_l = len(colors)-1

                for lineSegment in pointLineSegments:
                    pa = lineSegment[0]
                    pb = lineSegment[1]

                    if color_i == -1:
                        color = colors[0]
                    else:
                        color = colors[(color_i % color_l)+1]

                    cv2.line(frame, toplePoint(pa), toplePoint(pb), color, 2)

                    color_i += 1

                continue

            firstColor = (0, 0, 255)
            secondColor = (255, 0, 0)
            ic = 0

            for lineSegment in lines:
                pa = lineSegment[0]
                pb = lineSegment[1]

                if direction == 1:
                    cv2.line(frame, toplePoint(pa),
                             toplePoint(pb), firstColor, 2)
                else:
                    cv2.line(frame, toplePoint(pa),
                             toplePoint(pb), secondColor, 2)
                ic += 1

            # end for cont in contours
        cv2.line(frame, (10, 10), (15, 10), (0, 0, 255), 2)
        # end if not degugit

    else:
        cont = contours[ind % len(contours)]
        len_c = len(cont)
        if mind == -1:
            i = 0
            while i < len_c:
                pa = cont[i][0]
                pb = cont[(i+1) % len_c][0]
                cv2.line(frame, toplePoint(pa), toplePoint(pb), (0, 0, 255), 2)
                i += 1
            if printit:
                print('index ' + str(ind))
                print('=======')
        else:
            mind = mind % len_c
            pa = cont[mind][0]
            pb = cont[(mind+1) % len_c][0]
            cv2.line(frame, toplePoint(pa), toplePoint(pb), (0, 0, 255), 2)

            if printit:
                angle = math.atan2(pb[1]-pa[1], pb[0]-pa[0]) / math.pi*180
                print('mind ' + str(mind))
                print('pa ' + str(pa))
                print('pb ' + str(pb))
                print('ang ' + str(angle))
                print('dist ' + str(calc_distance(pa, pb)))
                print('=======')


def toplePoint(pa):
    return (pa[0], pa[1])


def checkfn():
    global ind, printit, mind, debugit
    doRescan = True
    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(']'):
            ind += 1
            doRescan = True

        elif key == ord('['):
            ind -= 1
            doRescan = True

        elif key == ord('p'):
            printit = True
            doRescan = True

        elif key == ord('o'):
            ind = poly_interest
            doRescan = True

        elif key == ord(','):
            mind -= 1
            doRescan = True

        elif key == ord('.'):
            mind += 1
            doRescan = True

        elif key == ord('/'):
            mind = -1
            doRescan = True

        elif key == ord('d'):
            debugit = not debugit
            doRescan = True

        if doRescan:
            frame = cv2.imread('sampleInput.png')
            qrcodescan(frame)
            frame = cv2.resize(
                frame, (frame.shape[1] * 2, frame.shape[0] * 2), interpolation=cv2.INTER_AREA)
            cv2.imshow('frame', frame)
            printit = False
            doRescan = False

    cv2.destroyAllWindows()


def scan():
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = cv2.VideoCapture(1)
            continue
        frameCopy = frame.copy()
        qrcodescan(frame)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            cv2.imwrite('sampleInput.png', frameCopy)

    cap.release()
    cv2.destroyAllWindows()


#checkfn()
scan()

exit
