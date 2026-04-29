import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    print("Frame status:", ret)

    if not ret:
        print("❌ Camera not working")
        break

    cv2.imshow("Test Cam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()