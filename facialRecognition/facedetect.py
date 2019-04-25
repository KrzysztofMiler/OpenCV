import cv2
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')#jak podjebie tutaj inne kaskady to je wykryje

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,#tu jest nasz img w grayscale
        scaleFactor=1.1,#jak twarze są blisko to się wydająduże i to ma na celu kompensacje tego
        minNeighbors=5,#is a parameter specifying how many neighbors each candidate rectangle should have, to retain it. A higher number gives lower false positives.
        minSize=(30, 30),# minimalny rozmiar okienka
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE#nwm czemu ale na tym się wywala
    )

    #Draw a rectangle around the faces
    for (x, y, w, h) in faces:#x,y to lewy górny róg gdzie wykryło twarz, (w,h) to są szerokość i wysokość twarzy, to (0,255,0) to kolor
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)# ten ost. to jak grube ramki

        roi_gray = gray[y:y+h, x:x+w]#niby to jest strefa w której rysujemy nasz prostokąt?
        roi_color = frame[y:y+h, x:x+w]#ale bez tego kody działa i nic się nie zmienia

        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(10, 10),
            #flags=cv2.cv.CV_HAAR_SCALE_IMAGE#nwm czemu ale na tym się wywala
        )
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()