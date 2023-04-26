import Image_stiching as ist 
import haris_corner as hcd 
import cv2

def corner_detection(video, fps):
    
    capture = cv2.VideoCapture(video)
    
    # getting video width and height
    width = int(capture.get(3))
    height = int(capture.get(4))
    
    output = cv2.VideoWriter("video_output.mp4",cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width,height))
 
    while(capture.isOpened()):
        ret, frame = capture.read()
        if (ret):
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = hcd.harris_corner_detector(gray_frame, 0.05, 0.8)
            keypoints, noCorners = hcd.keypoint_extraction(corners)
        
            print(noCorners)
            cv2.putText(frame, f'Corners Detected: {noCorners}', (10,1900), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
            for point in keypoints:
                cv2.circle(frame, point, 2, (255,0, 0))

            output.write(frame)
        else:
            break

    capture.release()
    output.release()
    cv2.destroyAllWindows()
   

if __name__ == "__main__":
    
    corner_detection("video.mp4",10)