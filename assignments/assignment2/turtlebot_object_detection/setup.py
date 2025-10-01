from setuptools import find_packages, setup

package_name = 'turtlebot_object_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/object_detection.launch.py']),
    ],
    install_requires=['setuptools', 'torch', 'torchvision', 'opencv-python', 'numpy', 'Pillow'],
    zip_safe=True,
    maintainer='vscode',
    maintainer_email='sohaibchachar12@gmail.com',
    description='Object detection package for TurtleBot3 using PyTorch COCO model',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'object_detection_node = turtlebot_object_detection.object_detection_node:main',
        ],
    },
)
