from setuptools import find_packages
from setuptools import setup

requirements = ["torch", "torchvision", "opencv-contrib-python"]

setup(
    name="yolo_human_counter",
    version="1.0",
    author="Lê Thu Hiền",
    author_email='lethuhienvtv99@gmail.com',
    description="Đồ án này mình chị gánh!",
    packages=find_packages(exclude=("configs", "tests",)),
    install_requires=requirements
)
