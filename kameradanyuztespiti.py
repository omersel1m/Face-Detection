import cv2

# Haarcascades dosyasının yolunu al
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

# Kamerayı başlat
cap = cv2.VideoCapture(0)  # 0, varsayılan kamera

while True:
    # Kameradan bir kare oku
    ret, frame = cap.read()
    if not ret:
        break
    
    # Görüntüyü aynala (yatay olarak)
    frame = cv2.flip(frame, 1)
    
    # Görüntüyü gri tonlara dönüştür
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Tespit edilen yüzlerin etrafına dikdörtgen çiz
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Tespit edilen yüzlerin gösterildiği pencereyi aç
    cv2.imshow('Detected Faces', frame)
    
    # Çıkış yapmak için 'q' tuşuna bas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
