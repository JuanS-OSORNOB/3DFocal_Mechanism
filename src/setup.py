import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="focmech3d",
    version="0.0.1",
    author="Juan Sebastián Osorno Bolívar & Amy Teegarden",
    author_email="focmech3d@gmail.com",
    description="Package for displaying 3D focal mechanisms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JuanS-OSORNOB/3DFocal_Mechanism",
    packages=setuptools.find_packages(exclude=["*.tests", "*.experiments"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering",
        "Framework :: Matplotlib"
    ],
    python_requires='>=3.6',
)