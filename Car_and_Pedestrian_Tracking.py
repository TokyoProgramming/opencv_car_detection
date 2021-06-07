import cv2


# img
img_file = 'car_img.png'
# video = cv2.VideoCapture('car_video.mp4')
# video = cv2.VideoCapture('car_video2.mp4')
video = cv2.VideoCapture('car_video4.mp4')

# pre-trained car classifier
classifier_file = 'car_detector.xml'

car_tracker = cv2.CascadeClassifier(classifier_file)

# Run forever until car stops or something
while True:
    
    # Read the current frame
    (read_successful, frame) = video.read()

    if read_successful:
        
        # Must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    else:         
        break
    
    # detect cars
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    
    # Draw rectangles around the cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, ( x,y), (x+w, y+h), (0,0,255),2)

    
    # Display the image with the faces spotted
    cv2.imshow('car Detector', frame)
    
    # don't autoclose (Wait here in the code and listen for a key press)
    cv2.waitKey(1)

# crete opencv image
img = cv2.imread(img_file)

# convert to grayscale 
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# track car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars => give coordinates
cars = car_tracker.detectMultiScale(black_n_white)

# Draw rectagles around the cars 
for(x,y,w,h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

# Display the image with the faces stoptted
cv2.imshow('programming', img)

# don't autoclose ( wait here in the code and listen for a key press)
cv2.waitKey()

print('code Completed')