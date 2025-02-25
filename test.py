import requests
import cv2

url = "http://localhost:8000/upload-video"


open("../input.mp4", "rb")
# lauch video

video = {"video": open("./input.mp4", "rb")}

data = {"type_of_request": 42}

response = requests.post(url, files=video, data=data)
print(response)
print(response.json())

# import cv2

# # Ouvrir la vidéo
# video = cv2.VideoCapture('../input.mp4')

# while True:
#     # Lire une frame
#     ret, frame = video.read()
    
#     # Quitter si la vidéo est terminée
#     if not ret:
#         break
    
#     # Afficher la frame
#     cv2.imshow('Video', frame)
    
#     # Quitter avec 'q'
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break

# # Libérer les ressources
# video.release()
# cv2.destroyAllWindows()