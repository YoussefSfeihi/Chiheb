from setuptools import setup

package_name = 'crater_dashboard'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/web', ['web/index.html']),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='Youssef Sfeihi',
    maintainer_email='sfeihiyoussef@gmail.com',
    description='Rover telemetry radio (UDP + ACK) and live Mission Control web dashboard for the C.R.A.T.E.R lunar scout',
    license='MIT',
    entry_points={'console_scripts': [
        'dashboard_server = crater_dashboard.dashboard_server:main',
        'telemetry_tx = crater_dashboard.telemetry_tx:main',
    ]},
)
