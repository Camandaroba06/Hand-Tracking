import cv2
import mediapipe as mp
video = cv2.VideoCapture(0)
hand = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils
Hand = hand.Hands(max_num_hands = 1)


while True:
    success, img = video.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultados = Hand.process(imgRGB)
    handspoints = resultados.multi_hand_landmarks
    if handspoints:
        for point in handspoints:
            print(point)
            mpDraw.draw_landmarks(img, point, hand.HAND_CONNECTIONS)
    cv2.imshow("Hand Tracking", img)
    cv2.waitKey(1)
