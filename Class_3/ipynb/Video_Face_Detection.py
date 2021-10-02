import cv2 as cv
#from cv2 import VideoWriter, VideoWriter_fourcc

detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

def detectImage(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    results = detector.detectMultiScale(gray, 1.5, 5)

    FaceID = 0
    for x, y, w, h in results:
        FaceID += 1
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(img, "Person_" + str(FaceID), (x - 10, y), cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 1)
    return img
    
def detectVideo():
    Video = cv.VideoCapture('D:\FirstTerm\computer_vision\homework\\3rd\\sing.mp4')
    frame_width = int(Video.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(Video.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(frame_width, frame_height)
    fps = 30
    video_writer = cv.VideoWriter('Video.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = Video.read()
        if not ret:
            break
        new_frame = detectImage(frame)

        # cv.imshow('img', new_frame)
        # cv.waitKey(0)

        video_writer.write(new_frame)
    
    video_writer.release()


if __name__=="__main__":
    detectVideo()