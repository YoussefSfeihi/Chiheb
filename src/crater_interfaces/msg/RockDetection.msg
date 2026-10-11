# Module 2 (Perception RGB-D) output for one detected object, in IMAGE space only.
# No global coordinates here: Module 3 back-projects (pixel_u, pixel_v, depth_m) with the
# camera intrinsics, tf2 and the rover pose.
string class_id        # "rock" (MVP)
float32 confidence     # YOLO score [0, 1]
float32 pixel_u        # bbox centre, column [px]
float32 pixel_v        # bbox centre, row [px]
float32 depth_m        # median depth over the bbox ROI [m]; NaN when depth_valid is false
bool depth_valid       # false: too few valid depth pixels (NaN/Inf/out of range) in the ROI
float32 bbox_w         # bbox width [px]  (lets Module 3 estimate the rock's size)
float32 bbox_h         # bbox height [px]
