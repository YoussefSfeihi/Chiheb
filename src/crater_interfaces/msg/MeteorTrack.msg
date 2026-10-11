# One incoming object tracked by the rover's RGB-D camera, in the `map` frame
# (map origin = rover start point; the station/lander is the `station` frame).
uint32 id
geometry_msgs/Point position          # latest fused position [m]
geometry_msgs/Vector3 velocity        # from the ballistic fit [m/s]
geometry_msgs/Point impact_point      # predicted ground impact [m]
float32 time_to_impact                # [s]; <0 if already impacted / not descending
float32 impact_sigma                  # 1-sigma uncertainty of the impact point [m]
float32 miss_distance                 # predicted impact distance from the rover at impact time [m]
bool is_threat                        # miss_distance < danger radius
uint32 n_obs
builtin_interfaces/Time first_seen
builtin_interfaces/Time last_seen
