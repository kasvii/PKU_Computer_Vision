import cv2 as cv

detector = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# 图像人脸检测
def detectImage(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    results = detector.detectMultiScale(gray, 1.5, 5)

    FaceID = 0
    for x, y, w, h in results:
        flag = 1
        # 删除检测到同一张人脸的框
        for x_, y_, w_, h_ in results:
            if x_==x and y_==y:
                continue
            if x > x_ and  y > y_ and x + w < x_ + w_ and y+ h < y_ + h_:
                flag=0
                break
        if flag == 0:
            continue
        
        FaceID += 1
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(img, "Person_" + str(FaceID), (x - 10, y), cv.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 1)
    return img
    
# 视频人脸检测    
def detectVideo():
    Video = cv.VideoCapture('D:\FirstTerm\computer_vision\homework\\3rd\\ad1080.mp4')
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
        video_writer.write(new_frame)

    video_writer.release()

if __name__=="__main__":
    detectVideo()