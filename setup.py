from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='SignFlow',
      version="0.0.1",
      description="Real-Time ASL Translation",
      license="MIT",
      author="J. Fenner, R. Kumar, E. Takeda, B. Lengereau",
      author_email="jaris.a.fenner@gmail.com",
      #url="https://github.com/Kaaykun/SignFlow",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
