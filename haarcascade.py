import cv2
import numpy as np


#oluşturmak istediğimiz sınıflamayı yani xml dosyasını yuz_casc nesnesine atadık.
yuz_casc=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")       #xml dosyasından bir CascadeClassifier fonksiyonundan bir obje üretiyor.xml dosyasındaki algoritmayı kullanıyor.

 #resmi okuduk.
resim=cv2.imread("ataturk.jpg")

#CascadeClassifier() fonksiyonundan algoritmayı kullanabilmem için resmi gri tona getirmem gereklidir.
gri_ton=cv2.cvtColor(resim,cv2.COLOR_BGR2GRAY) 

#detectMultiScale() fonksiyonu yüzleri tespit ediyor ve bunları bir liste şeklinde döndürüyor.
#ilk parametre gri tonlama aralığındaki objesini istiyor.
#ikinci parametre de resmin ne kadar büyütüleceğini seçiyoruz.
#3.parametre kontrol aşaması. resmi ne kadar tarayacağını belirtiriz.
#yuzler adlı değişkenimiz liste döndürüyor.
yuzler=yuz_casc.detectMultiScale(gri_ton,1.1,4)

#buradan çıkan(console de) iç içe listede ilk parametresi dikdörtgene aldığı alanın sol üst kısmındaki x değeri
#ikinci paarmetresi ise y değeri üçüncü parametresi alınan alanın genişliği 4.sü ise alınan alanın yüksekliği
print(yuzler)      

#resimde birden fazla yüz olabileceğinden for döngüsüne aldık.
for (x,y,w,h) in yuzler:
    cv2.rectangle(resim,(x,y),(x+w,y+h),(255,0,0),2)        
    #ilk parametre hangi resimde dikdörtgen yapacağımızı
    #ikinci parametre çizilecek seçilecek alanın sol üst köşesinin x y verir. 3. parametre seçilen alanın sağ alt 
    #köşesini verir. 4. parametre BGR kodları(çizilecek dörtgenin rengi için) sonuncusu çizilen dikdörtgenin kalınlığı

cv2.imshow("resim",resim)

cv2.waitKey(0)
cv2.destroyAllWindows()
