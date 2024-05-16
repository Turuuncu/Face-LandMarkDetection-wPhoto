import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Models/shape_predictor_68_face_landmarks.dat")



def detect_facial_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    landmarks = []

    for face in faces:
        shape = predictor(gray, face)
        
        for i in range(68):
            x = shape.part(i).x
            y = shape.part(i).y
            landmarks.append((x, y))

    return landmarks

image_path = "Image/Path"
image = cv2.imread(image_path)

landmarks = detect_facial_landmarks(image)


for landmark in landmarks:
    cv2.circle(image, landmark, 1, (0, 0, 255), -1)

cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
