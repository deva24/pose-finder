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

    fromAngle = -30
    toAngle = 30

    if not debugit:
        for cont in contours:
            len_c = len(cont)

            aLine = []
            foundHorizontalLine = False

            i = int(len_c * 0.4)
            loop_over_len = int(len_c*1.6)

            fromAngle = -30
            toAngle = 30

            attempt = 0
            while attempt < 2:
                while i < loop_over_len:
                    pa = cont[(i-1) % len_c][0]
                    pb = cont[i% len_c][0]
                    pc = cont[(i+1) % len_c][0]
                    pd = cont[(i+2) % len_c][0]

                    angle_ab = math.atan2(pb[1]-pa[1], pb[0]-pa[0])/math.pi*180
                    angle_bc = math.atan2(pc[1]-pb[1], pc[0]-pb[0])/math.pi*180
                    angle_cd = math.atan2(pd[1]-pc[1], pd[0]-pc[0])/math.pi*180

                    dis_ab = distance(pa, pb)
                    dis_bc = distance(pb, pc)
                    dis_cd = distance(pc, pd)

                    lookslikeLine = False

                    if dis_bc > dis_ab and dis_bc > dis_cd:
                        if angle_bc >= fromAngle and angle_bc <= toAngle:
                            lookslikeLine = True
                    elif dis_ab > dis_bc and dis_cd > dis_bc:
                        if angle_ab >= fromAngle and angle_ab <= toAngle and angle_cd >= fromAngle and angle_cd <= toAngle:
                            lookslikeLine = True

                    if lookslikeLine:
                        aLine.append(i)
                        
                    else:
                        if len(aLine) > 0:
                            lineIndexBegin = aLine[0]
                            lineIndexEnd = aLine[len(aLine)-1] + 1

                            pa = cont[lineIndexBegin % len_c][0]
                            pb = cont[lineIndexEnd % len_c][0]
                            if distance(pa, pb) > 5:
                                foundHorizontalLine=True
                                break
                            else:
                                aLine=[]
                    i += 1
                if foundHorizontalLine:
                    # line stores index i of a line consisting of point[i] and point[i+1] in contour
                    lineIndexBegin = aLine[0]
                    lineIndexEnd = aLine[len(aLine)-1] + 1

                    pa = cont[lineIndexBegin % len_c][0]
                    pb = cont[lineIndexEnd % len_c][0]

                    if distance(pa, pb) > 5:
                        if attempt == 0:
                            cv2.line(frame, toplePoint(pa),
                                    toplePoint(pb), (0, 0, 255), 2)
                        else:
                            cv2.line(frame, toplePoint(pa),
                                    toplePoint(pb), (255, 0, 0), 2)
                        break

                fromAngle-=90
                toAngle-=90
                attempt+=1
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
            ind = 72
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
            frame = cv2.resize(frame,(frame.shape[1] * 2,frame.shape[0] * 2),interpolation=cv2.INTER_AREA)
            cv2.imshow('frame', frame)
            printit=False
            doRescan=False

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