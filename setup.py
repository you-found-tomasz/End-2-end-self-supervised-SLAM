from setuptools import setup,find_packages

setup(name='end2endslam',
      version='0.1',
      packages=find_packages(exclude=("docs", "test", "examples","gradslam")),
      zip_safe=True,
      include_dirs=[])