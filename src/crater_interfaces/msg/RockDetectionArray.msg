# All rock detections of one RGB-D frame. header = the RGB frame's stamp and optical frame
# (camera_optical). An empty array is an explicit "no rock in this frame".
std_msgs/Header header
RockDetection[] detections
