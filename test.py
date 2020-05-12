import numpy as np
import cv2
import math


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


def qrcodescan(frame):
    global ind, printit, mind, debugit
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    thr = math.floor(frame.mean())
    ret, thresh = cv2.threshold(gray, thr, 255, 0)
    thresh = cv2.blur(thresh, (3, 3))
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not debugit:
        for cont in contours:
            len_c = len(cont)

            predictedLine = []
            aLine = []
            isNewLine = True

            i = 0
            while i < len_c:
                pa = cont[(i-1) % len_c][0]
                pb = cont[i][0]
                pc = cont[(i+1) % len_c][0]
                pd = cont[(i+2) % len_c][0]

                angle_ab = math.atan2(pb[1]-pa[1], pb[0]-pa[0])/math.pi*180
                angle_bc = math.atan2(pc[1]-pb[1], pc[0]-pb[0])/math.pi*180
                angle_cd = math.atan2(pd[1]-pc[1], pd[0]-pc[0])/math.pi*180

                dis_ab = distance(pa, pb)
                dis_bc = distance(pb, pc)
                dis_cd = distance(pc, pd)

                fromAngle = -30
                toAngle = 30

                lookslikeLine = False

                if dis_bc > dis_ab and dis_bc > dis_cd:
                    if angle_bc >= fromAngle and angle_bc <= toAngle:
                        lookslikeLine = True
                elif dis_ab > dis_bc and dis_cd > dis_bc:
                    if angle_ab >= fromAngle and angle_ab <= toAngle and angle_cd >= fromAngle and angle_cd <= toAngle:
                        lookslikeLine = True

                if lookslikeLine:
                    if not isNewLine:
                        isNewLine = True
                    aLine.append(i)

                elif isNewLine:
                    if len(aLine) > 0:
                        predictedLine.append(aLine)
                        isNewLine = False
                        aLine = []

                i += 1

            for line in predictedLine:
                # line stores index i of a line consisting of point[i] and point[i+1] in contour
                lineIndexBegin = line[0]
                lineIndexEnd = line[len(line)-1] + 1

                pa = cont[lineIndexBegin][0]
                pb = cont[lineIndexEnd % len_c][0]

                if distance(pa,pb)>5:
                    cv2.line(frame, toplePoint(pa), toplePoint(pb), (0, 0, 255), 2)
                

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
        else:
            mind = mind % len_c
            pa = cont[mind][0]
            pb = cont[(mind+1) % len_c][0]
            cv2.line(frame, toplePoint(pa), toplePoint(pb), (0, 0, 255), 2)

            if printit:
                angle = math.atan2(pb[1]-pa[1], pb[0]-pa[0]) / math.pi*180
                print('pa ' + str(pa))
                print('pb ' + str(pb))
                print('ang ' + str(angle))
                print('dist ' + str(distance(pa, pb)))
                print('\n')


def toplePoint(pa):
    return (pa[0], pa[1])


def checkfn():
    global ind, printit, mind, debugit

    frame = cv2.imread('sampleInput.png')
    qrcodescan(frame)

    cv2.imshow('frame', frame)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(']'):
            ind += 1
            frame = cv2.imread('sampleInput.png')
            qrcodescan(frame)
            cv2.imshow('frame', frame)

        elif key == ord('['):
            ind -= 1
            frame = cv2.imread('sampleInput.png')
            qrcodescan(frame)
            cv2.imshow('frame', frame)

        elif key == ord('p'):
            printit = True
            frame = cv2.imread('sampleInput.png')
            qrcodescan(frame)
            printit = False
            cv2.imshow('frame', frame)

        elif key == ord('o'):
            ind = 172
            frame = cv2.imread('sampleInput.png')
            qrcodescan(frame)
            cv2.imshow('frame', frame)

        elif key == ord(','):
            mind -= 1
            frame = cv2.imread('sampleInput.png')
            qrcodescan(frame)
            cv2.imshow('frame', frame)

        elif key == ord('.'):
            mind += 1
            frame = cv2.imread('sampleInput.png')
            qrcodescan(frame)
            cv2.imshow('frame', frame)

        elif key == ord('/'):
            mind = -1
            frame = cv2.imread('sampleInput.png')
            qrcodescan(frame)
            cv2.imshow('frame', frame)
        elif key == ord('d'):
            debugit = not debugit
            frame = cv2.imread('sampleInput.png')
            qrcodescan(frame)
            cv2.imshow('frame', frame)

    cv2.destroyAllWindows()


def scan():
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
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


# checkfn()
scan()
