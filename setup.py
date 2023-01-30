import os
from setuptools import setup

os.environ['PROJECT_DIR'] = os.getcwd()

setup (
  name='covid19framing',
  version='0.1',
  packages=[''],
  description="Adapting Topic Modeling Techniques for Interpreting Framings of the COVID-19 Pandemic",
  url='https://github.com/cmmattingly/covid19framing',
  author='Chase Mattingly',
  author_email='chase.mattingly68@gmail.com',
  license='Lehigh',
  install_requires=[
    '',
  ],
)