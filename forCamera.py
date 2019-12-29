import cv2
import numpy as np

kamera=cv2.VideoCapture(0)

fourcc= cv2.VideoWriter_fourcc(*'XVID')     #alınan videoyu kayıt etmek için

kayit=cv2.VideoWriter('kayit.avi',fourcc,20,(640,480))  #alınan videoyu kayıt etmek için

while True:
    ret,kare=kamera.read()

    yuz_casc=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")   

    gri_ton=cv2.cvtColor(kare,cv2.COLOR_BGR2GRAY) 

    yuzler=yuz_casc.detectMultiScale(gri_ton,1.1,4)    

    for (x,y,w,h) in yuzler:
        cv2.rectangle(kare,(x,y),(x+w,y+h),(255,0,0),2)        

    kayit.write(kare)       #kayit edilen videoyu yazdırmak için(yani klasör olarak ekler)

    cv2.imshow("kamera",kare)

    if cv2.waitKey(25) & 0xFF == ord('q'):      #25 milisaniyede bir tekrar eksin ve q basınca çıksın
        break
kamera.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
