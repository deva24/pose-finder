import numpy as np
import cv2
import math

poly_interest = 99


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


def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def angle1(a, b):
    return math.atan2(b[1]-b[0], a[1]-a[0])*180/math.pi


def deviation(a, b, c):
    ab = distance(a, b)
    bc = distance(b, c)
    ac = distance(a, c)

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


debugit = False
ind = 0
mind = -1
printit = False

def findLines(cont):
    i = 0
    len_c = len(cont)

    predictedLines = []
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

    while i < len_c:
        if type(pa) == type(None):
            pa = cont[i % len_c][0]
        if type(pb) == type(None):
            pb = cont[(i+1) % len_c][0]
        if type(pc) == type(None):
            pc = cont[(i+2) % len_c][0]

        if type(dis_ab) == type(None):
            dis_ab = distance(pa, pb)
        if type(dis_bc) == type(None):
            dis_bc = distance(pb, pc)
        if type(dis_ac) == type(None):
            dis_ac = distance(pa, pc)

        # if type(ang_ab) == type(None):
        #     ang_ab = angle1(pa, pb)
        # if type(ang_bc) == type(None):
        #     ang_bc = angle1(pb, pc)
        # if type(ang_ac) == type(None):
        #     ang_ac = angle1(pa, pc)

        didMerge = False

        if dis_ab + dis_bc - dis_ac <= 0.5:
            didMerge = True

        if didMerge:
            if not lineWasFound:
                predictedLine = [(i) % len_c, (i+1) % len_c]
                predictedLines.append(predictedLine)
            else:
                predictedLine[1] = (i+1) % len_c

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

            # ang_ab = ang_ac
            # ang_bc = None
            # ang_ac = None

            lineWasFound = True

        else:
            # remove predicted line if its too short
            if type(predictedLine) != type(None):
                pl1 = cont[predictedLine[0]%len_c][0]
                pl2 = cont[(predictedLine[1]+1)%len_c][0]

                if distance(pl1,pl2) < 5:
                    predictedLines.pop()
                predictedLine=None

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

            # ang_ab = ang_bc
            # ang_bc = None
            # ang_ac = None

            lineWasFound = False
            # end if
        i += 1
        # end while i < len_cc
    return predictedLines


def qrcodescan(frame):
    global ind, printit, mind, debugit
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    thr = math.floor(frame.mean()*2/3)
    ret, thresh = cv2.threshold(gray, thr, 255, 0)
    thresh = cv2.blur(thresh, (3, 3))
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #contours = [contours[poly_interest]]

    if not debugit:
        for cont in contours:
            predictedLines = findLines(cont)

            len_c = len(cont)
            colors = [ (0, 175, 255), (0, 180, 0),(255,255,0), (255, 0, 0)]
            firstColor =(0, 0, 255)
            ic = 0
            lc = len(colors)
            doFirstColor=True
            for lineIndices in predictedLines:
                pa = cont[lineIndices[0] % len_c][0]
                pb = cont[(lineIndices[1]+1) % len_c][0]
                if doFirstColor:
                    doFirstColor=False

                    cv2.line(frame, toplePoint(pa),
                         toplePoint(pb), firstColor, 2)
                else:
                    cv2.line(frame, toplePoint(pa),
                         toplePoint(pb), colors[ic % lc], 2)
                ic += 1

            # end for cont in contours
        cv2.line(frame,(10,10),(15,10),(0,0,255),2)
        # end if not degugit

    else:
        cont = contours[ind]
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
                print('dist ' + str(distance(pa, pb)))
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


checkfn()
#scan()

exit
