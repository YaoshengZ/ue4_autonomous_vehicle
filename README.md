# UE4_Autonomous_Vehicle
This project is based on Tronis (UE4) simulation environment with OpenCV and implements some ADAS funtions.
## Demo vidow
![Watch the video](/demo.mp4 "demo video")
## Blue Print Settings
![Alt text](/blueprint_settings/blueprint_settings1.png "blueprint settings 1")
![Alt text](/blueprint_settings/blueprint_settings2.png "blueprint settings 2")
## Red Traffic Light Detection
1. HLS detector: The red pixel on the tronis image will be filtered. The hls parameters that determine the range that how "red" the red light can be could be adjusted base on environment.
2. ROI crop: Crop out the region of the image that the where the traffic light could appear when the car is moving. 
3. Red traffic light roi: Only this certain region of image will be detected if there is a red traffic light.
4. Stop at traffic light: This step will be implemented in throttle control section.
## Lane Detection
1. Edge detector: First convert the image to gaussian blurred image then to gray scale image. The binary conversion can be implemented depends on environment, in this case no binary conversion needed. Then use Canny edge detection to find all edges. In this case, canny outperform sobel. For better hough line detection later, the edges will be dilated.
2. HLS detector for white and yellow: The lane lines are in general white or yellow. If there are only white or yellow lines in the environment, using a white or yellow hls filter and then make a fusion with edge detection result could make the detection robuster. Another option is to combine the results of white and yellow hls filter if there both white and yellow lines. Also, the result will be dilated.
3. ROI crop: To avoid interference of the objects outside of the lane, only the roi region will be kept for hough line detection.
4. Hough line detection: To detect lane lines on ROI image.
5. Lane display: The detected lines will be divided into left and right lane lines, then calculate the middle line of the lane.
This lane detection result is robust for sunlight reflexion problem as long as the situation is not too extrem.
This lane detection result is also stable for a long enough distance if there is only one lane line, no matter it is left or right line, except at the starting position.
## Steering Control
Here the PID controller will be implemented. In order to have a smoother and quicker reaction, the target point for PID control will be upper point of the middle line, which is couple meters ahead of the vehicle.
## Throttle Control
The deteciton of other moving or static vehicles is based on bounding box detector in Tronis. If there is no vehicle within acc activation distance, only throttle PID controller will be activated. If there are vehicles within acc activation distance, the distance PID controller will also be activated. When the detected red pixel in the roi reach a certain threshold, it will be considered that a red traffic light is detected. If the current speed is above a certain threshold, the vehicle will brake, and the throttle input will be set to 0 when the speed is low enough.
