from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='rydberg',
      version='0.0.1',
      description='Package with tools useful for analysing NH3-Rydberg atom collisions, including a simulation and ramsey fit.',
      url='',
      author='Alex Morgan',
      author_email='axm108@gmail.com',
      license='BSD 3-clause',
      packages=['nh3_collisions'],
      include_package_data=True,
      zip_safe=False)