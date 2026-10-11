from glob import glob

from setuptools import setup

package_name = 'crater_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models', glob('models/*.onnx')),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='Youssef Sfeihi',
    maintainer_email='sfeihiyoussef@gmail.com',
    description='Onboard RGB-D perception (Module 2: rock detection + depth fusion), meteor tracking, station-relative navigation and evasion for the C.R.A.T.E.R lunar scout',
    license='MIT',
    entry_points={
        'console_scripts': [
            'meteor_tracker = crater_perception.meteor_tracker:main',
            'mission_controller = crater_perception.mission_controller:main',
            'rgbd_sync_node = crater_perception.rgbd_sync_node:main',
            'rock_detector_node = crater_perception.rock_detector_node:main',
            'depth_fusion_node = crater_perception.depth_fusion_node:main',
            'rock_localizer = crater_perception.rock_localizer:main',
        ],
    },
)
