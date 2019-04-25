import cv2
import os
cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
while(True):
    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        path = 'C:/Users/21st C/Documents/OpenCV/facialRecognition/dataset' #do jakiego folderu zapisywać
                                                                                                #wybór rozmiaru by całego pokoju nie zapisać
        cv2.imwrite(os.path.join(path , 'User.'+ str(face_id) + '.' + str(count) + ".jpg"), gray[y:y+h,x:x+w])#te czarnobiałe zapisuje bo ich do treningu potrzebuje
        cv2.imshow('image', img)
    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 50: # TUTAJ USTAWIAJ ile zdj ma zrobić
         break

print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()