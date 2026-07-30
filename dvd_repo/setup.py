import os

from setuptools import find_packages, setup

# Path to the requirements file
requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

# Read the requirements manually without pkg_resources
install_requires = []
if os.path.exists(requirements_path):
    with open(requirements_path, 'r', encoding='utf-8') as f:
        for line in f:
            req = line.strip()
            # 忽略空行和以 '#' 开头的注释行
            if req and not req.startswith('#'):
                install_requires.append(req)

setup(
    name="DVD",
    version="1.0.0",
    author="Artiprocher",
    packages=find_packages(),
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    package_data={"diffsynth": ["tokenizer_configs/**/**/*.*"]},
    python_requires='>=3.6',
)


