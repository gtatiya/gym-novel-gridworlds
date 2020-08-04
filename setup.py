from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='gym-novel-gridworlds',
      version='1.2',
      install_requires=['gym', 'matplotlib', 'numpy', 'keyboard'],
      author="Gyan Tatiya",
      author_email="Gyan.Tatiya@tufts.edu",
      description="Gym Novel Gridworlds are environments for OpenAI Gym",
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://github.com/gtatiya/gym-novel-gridworlds",
      license='MIT License',
      python_requires='>=3.6',
      )
