"""
crater_sim.launch.py: C.R.A.T.E.R lunar scout under meteor bombardment.

Gazebo (open lunar plain)  ->  ros_gz_bridge  ->  wheel+IMU EKF (dead reckoning)
  -> Nav2  <-  mission_controller (patrol around the station, print station-relative
                                   position, evade incoming meteors)
  RGB-D camera -> meteor_tracker (ballistic fit, impact prediction) -> mission_controller
  RGB-D camera -> Module 2 perception: rgbd_sync_node -> rock_detector_node (YOLO)
                  -> depth_fusion_node -> /perception/rocks -> rock_localizer (Module 3) -> Nav2
  meteor_spawner: the space environment (simulation side, uses ground truth)

Usage:
  ros2 launch crater_sim crater_sim.launch.py
  ros2 launch crater_sim crater_sim.launch.py meteor_period:=15
  ros2 launch crater_sim crater_sim.launch.py headless:=true rviz:=false
  Mission Control dashboard: open http://localhost:8080 in a browser
  Harsher radio: ros2 launch crater_sim crater_sim.launch.py link_loss:=0.3 link_delay_ms:=250
"""

import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, GroupAction, IncludeLaunchDescription,
    SetEnvironmentVariable, TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue

# Sensor mounts relative to base_link: x y z roll pitch yaw.
# MUST match the <pose> of the sensors in models/rover/model.sdf.
SENSOR_TF = {
    'lidar':          (0.30, 0.0, 0.35, 0.0, 0.0, 0.0),
    'camera_link':    (0.35, 0.0, 0.50, 0.0, -0.2, 0.0),   # tilted 11 deg up
    'base_footprint': (0.0, 0.0, -0.23, 0.0, 0.0, 0.0),
}
STATION_XY = (-2.6, 0.0)   # lander position in map (see tools/gen_world.py STATION_POS)
AI_PYTHON = os.path.expanduser('~/crater_ws/.venv-ai/bin/python')   # see tools/train_yolo.sh


def static_tf(parent, child, x, y, z, roll, pitch, yaw):
    return Node(
        package='tf2_ros', executable='static_transform_publisher', name=f'tf_{child}',
        arguments=['--x', str(x), '--y', str(y), '--z', str(z),
                   '--roll', str(roll), '--pitch', str(pitch), '--yaw', str(yaw),
                   '--frame-id', parent, '--child-frame-id', child],
        parameters=[{'use_sim_time': True}])


def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_sim = get_package_share_directory('crater_sim')
    nav2_params_file = os.path.join(pkg_sim, 'config', 'nav2_lunar_params.yaml')
    ekf_params_file = os.path.join(pkg_sim, 'config', 'ekf_odom.yaml')
    rviz_config_file = os.path.join(pkg_sim, 'config', 'crater.rviz')
    # Gazebo GUI layout: default plugins + AI camera panel (top-left)
    gui_config = os.path.join(pkg_sim, 'config', 'gazebo_gui.config')

    args = [
        DeclareLaunchArgument('world', default_value='lunar_crater.sdf',
                              description='World file inside crater_sim/worlds, or an absolute path'),
        DeclareLaunchArgument('headless', default_value='false', description='Run Gazebo without GUI'),
        DeclareLaunchArgument('rviz', default_value='true', description='Launch RViz2'),
        DeclareLaunchArgument('meteors', default_value='true', description='Throw meteors at the rover'),
        DeclareLaunchArgument('meteor_period', default_value='20.0', description='Mean seconds between meteors'),
        DeclareLaunchArgument('meteor_first_delay', default_value='35.0', description='Seconds before the first meteor'),
        DeclareLaunchArgument('dashboard', default_value='true',
                              description='Serve the Mission Control web dashboard'),
        DeclareLaunchArgument('dashboard_port', default_value='8080', description='Dashboard HTTP port'),
        DeclareLaunchArgument('telemetry_port', default_value='5005', description='Ground station UDP port'),
        DeclareLaunchArgument('link_loss', default_value='0.10', description='Emulated radio: drop probability'),
        DeclareLaunchArgument('link_delay_ms', default_value='80.0', description='Emulated radio: one-way delay'),
        DeclareLaunchArgument('link_jitter_ms', default_value='40.0', description='Emulated radio: +- jitter'),
        DeclareLaunchArgument('link_corrupt', default_value='0.01', description='Emulated radio: bit-flip probability'),
        DeclareLaunchArgument('perception', default_value='true',
                              description='Module 2 RGB-D perception (sync + YOLO + depth fusion) + rock '
                                          'localizer; annotated camera POV in the Gazebo top-left panel'),
        DeclareLaunchArgument('conf_threshold', default_value='0.35', description='YOLO confidence threshold'),
        DeclareLaunchArgument('rocks_model', default_value=os.path.join(
            get_package_share_directory('crater_perception'), 'models', 'rock_yolo.onnx'),
            description='YOLOv8 ONNX model (tools/train_yolo.sh)'),
        DeclareLaunchArgument('sensor_viz', default_value='true',
                              description='Draw the RGB-D camera frustum + scan pulse in Gazebo'),
        DeclareLaunchArgument('patrol', default_value='true',
                              description='Autonomous patrol (false: send Nav2 goals from RViz yourself)'),
    ]
    world = PathJoinSubstitution([pkg_sim, 'worlds', LaunchConfiguration('world')])
    headless = LaunchConfiguration('headless')

    # =====================================================================
    #  1. GAZEBO (models + gz Python systems on the search paths)
    # =====================================================================
    env = [
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.pathsep.join(
            [os.path.join(pkg_sim, 'models'), os.environ.get('GZ_SIM_RESOURCE_PATH', '')])),
        SetEnvironmentVariable('PYTHONPATH', os.pathsep.join(
            [os.path.join(pkg_sim, 'gz_systems'), os.environ.get('PYTHONPATH', '')])),
    ]
    gz_sim_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-r -s --headless-rendering ', world]}.items(),
        condition=IfCondition(headless))
    gz_sim_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': ['-r --gui-config ', gui_config, ' ', world]}.items(),
        condition=UnlessCondition(headless))

    # =====================================================================
    #  2. GAZEBO <-> ROS BRIDGE
    # =====================================================================
    bridge = TimerAction(period=5.0, actions=[Node(
        package='ros_gz_bridge', executable='parameter_bridge', name='gz_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/wheel/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            # RGB-D camera (gz topics /camera/*; renamed on the ROS side below)
            '/camera/image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            # annotated AI view, ROS -> Gazebo (shown in the GUI's top-left ImageDisplay panel)
            '/perception/debug_image@sensor_msgs/msg/Image]gz.msgs.Image',
            # ground-truth boxes: YOLO training/evaluation only, never used onboard
            '/camera/bbox@vision_msgs/msg/Detection2DArray[gz.msgs.AnnotatedAxisAligned2DBox_V',
            # evaluation / environment only (never consumed by onboard nodes)
            '/ground_truth/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        ],
        # Module 2 interface names (ATEC'26 perception spec)
        remappings=[('/camera/image', '/camera/rgb/image_raw'),
                    ('/camera/depth_image', '/camera/depth/image_raw')],
        parameters=[{'use_sim_time': True}],
        output='screen')])

    # =====================================================================
    #  3. STATIC TF: sensor mounts, optical frame, map == odom, station
    # =====================================================================
    tfs = [static_tf('base_link', child, *pose) for child, pose in SENSOR_TF.items()]
    tfs.append(static_tf('camera_link', 'camera_optical', 0, 0, 0, -math.pi / 2, 0, -math.pi / 2))
    # Open plain, no SLAM: navigation frame = EKF odometry frame (dead reckoning).
    tfs.append(static_tf('map', 'odom', 0, 0, 0, 0, 0, 0))
    tfs.append(static_tf('map', 'station', STATION_XY[0], STATION_XY[1], 0, 0, 0, 0))

    # =====================================================================
    #  4. ODOMETRY: wheel vx + IMU gyro EKF -> /odom, TF odom->base_link
    # =====================================================================
    ekf = TimerAction(period=6.0, actions=[Node(
        package='robot_localization', executable='ekf_node', name='ekf_odom', output='screen',
        parameters=[ekf_params_file], remappings=[('odometry/filtered', '/odom')])])

    # =====================================================================
    #  5. NAV2 (minimal set)
    # =====================================================================
    nav2_remappings = [('/tf', 'tf'), ('/tf_static', 'tf_static')]
    nav2_nodes = [
        ('nav2_controller', 'controller_server', [('cmd_vel', 'cmd_vel_nav')]),
        ('nav2_smoother', 'smoother_server', []),
        ('nav2_planner', 'planner_server', []),
        ('nav2_behaviors', 'behavior_server', [('cmd_vel', 'cmd_vel_nav')]),
        ('nav2_bt_navigator', 'bt_navigator', []),
        ('nav2_velocity_smoother', 'velocity_smoother', [('cmd_vel', 'cmd_vel_nav')]),
        ('nav2_collision_monitor', 'collision_monitor', []),
    ]
    nav2 = TimerAction(period=10.0, actions=[GroupAction(
        actions=[SetParameter('use_sim_time', True)] + [
            Node(package=pkg, executable=exe, name=exe, output='log',
                 parameters=[nav2_params_file], remappings=nav2_remappings + extra)
            for pkg, exe, extra in nav2_nodes
        ])])
    # Started 6 s after the Nav2 servers: if it calls their lifecycle services before DDS
    # discovery has finished (likely under CPU load), the replies are lost and bring-up hangs.
    nav2_lifecycle = TimerAction(period=16.0, actions=[Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation', output='log',
        parameters=[{'use_sim_time': True, 'autostart': True, 'bond_timeout': 10.0,
                     'node_names': [n[1] for n in nav2_nodes]}])])

    # =====================================================================
    #  6. ONBOARD: RGB-D meteor tracker + mission controller
    # =====================================================================
    # =====================================================================
    #  6c. MODULE 2 PERCEPTION RGB-D + Module 3 rock localizer + Gazebo sensor visualization
    #      /camera/{rgb,depth} -> rgbd_sync_node -> rock_detector_node (YOLO)
    #        -> /perception/detections_2d -> depth_fusion_node -> /perception/rocks
    #        -> rock_localizer -> /perception/rock_points -> Nav2 hazard_layer
    # =====================================================================
    perception_on = IfCondition(LaunchConfiguration('perception'))
    ai = TimerAction(period=10.0, actions=[
        Node(package='crater_perception', executable='rgbd_sync_node', name='rgbd_sync_node', output='screen',
             condition=perception_on, parameters=[{'use_sim_time': True}]),
        # runs with the runtime venv's interpreter: system ROS packages + onnxruntime
        # (OpenCV 4.6's DNN module cannot load YOLOv8 graphs)
        Node(package='crater_perception', executable='rock_detector_node', name='rock_detector_node',
             output='screen', prefix=AI_PYTHON, condition=perception_on,
             parameters=[{'use_sim_time': True, 'model_path': LaunchConfiguration('rocks_model'),
                          'conf_threshold': ParameterValue(LaunchConfiguration('conf_threshold'), value_type=float)}]),
        Node(package='crater_perception', executable='depth_fusion_node', name='depth_fusion_node', output='screen',
             condition=perception_on, parameters=[{'use_sim_time': True}]),
        Node(package='crater_perception', executable='rock_localizer', name='rock_localizer', output='screen',
             condition=perception_on, parameters=[{'use_sim_time': True}]),
        Node(package='crater_sim', executable='sensor_viz.py', name='sensor_viz', output='screen',
             condition=IfCondition(PythonExpression(["'", LaunchConfiguration('sensor_viz'), "' == 'true' and '",
                                                     headless, "' != 'true'"])),
             parameters=[{'use_sim_time': True}]),
    ])

    onboard = TimerAction(period=12.0, actions=[
        Node(package='crater_perception', executable='meteor_tracker', name='meteor_tracker',
             output='screen', parameters=[{'use_sim_time': True}]),
        Node(package='crater_perception', executable='mission_controller', name='mission_controller',
             output='screen', condition=IfCondition(LaunchConfiguration('patrol')),
             parameters=[{'use_sim_time': True, 'station_x': STATION_XY[0], 'station_y': STATION_XY[1]}]),
    ])

    # =====================================================================
    #  6b. RADIO LINK + MISSION CONTROL DASHBOARD
    #      rover telemetry_tx --UDP datagrams, each ACKed--> ground dashboard_server
    #      --SSE--> http://localhost:<dashboard_port>
    # =====================================================================
    link_params = {
        'link_loss': ParameterValue(LaunchConfiguration('link_loss'), value_type=float),
        'link_delay_ms': ParameterValue(LaunchConfiguration('link_delay_ms'), value_type=float),
        'link_jitter_ms': ParameterValue(LaunchConfiguration('link_jitter_ms'), value_type=float),
        'link_corrupt': ParameterValue(LaunchConfiguration('link_corrupt'), value_type=float),
    }
    telemetry_port = ParameterValue(LaunchConfiguration('telemetry_port'), value_type=int)
    dashboard = TimerAction(period=12.0, actions=[GroupAction(
        condition=IfCondition(LaunchConfiguration('dashboard')),
        actions=[
            Node(package='crater_dashboard', executable='telemetry_tx', name='telemetry_tx', output='screen',
                 parameters=[dict(link_params, use_sim_time=True, ground_host='127.0.0.1',
                                  ground_port=telemetry_port, station_x=STATION_XY[0], station_y=STATION_XY[1])]),
            Node(package='crater_dashboard', executable='dashboard_server', name='dashboard_server', output='screen',
                 parameters=[dict(link_params, use_sim_time=True, telemetry_port=telemetry_port,
                                  port=ParameterValue(LaunchConfiguration('dashboard_port'), value_type=int),
                                  station_x=STATION_XY[0], station_y=STATION_XY[1])]),
        ])])

    # =====================================================================
    #  7. SPACE ENVIRONMENT: meteor spawner (simulation side)
    # =====================================================================
    spawner = TimerAction(period=8.0, actions=[Node(
        package='crater_sim', executable='meteor_spawner.py', name='meteor_spawner', output='screen',
        condition=IfCondition(LaunchConfiguration('meteors')),
        parameters=[{'use_sim_time': True, 'period': LaunchConfiguration('meteor_period'),
                     'first_delay': LaunchConfiguration('meteor_first_delay')}])])

    # =====================================================================
    #  8. RVIZ2 (software rendering env vars only on RViz: they break Gazebo ogre2)
    # =====================================================================
    rviz = TimerAction(period=10.0, actions=[ExecuteProcess(
        cmd=['rviz2', '-d', rviz_config_file, '--ros-args', '-p', 'use_sim_time:=true'],
        output='log',
        additional_env={'LIBGL_ALWAYS_SOFTWARE': '1', 'MESA_GL_VERSION_OVERRIDE': '3.3',
                        'QT_QUICK_BACKEND': 'software', 'OGRE_RTT_MODE': 'Copy'},
        condition=IfCondition(LaunchConfiguration('rviz')))])

    return LaunchDescription(args + env + [gz_sim_headless, gz_sim_gui, bridge, ekf] + tfs + [
        nav2, nav2_lifecycle, onboard, ai, dashboard, spawner, rviz])
