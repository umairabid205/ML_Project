# This file will be responsible for in create my Machine Learning application as a package
# I'll be abble to build my ML application as a package and even deploy it to PyPI from there anybody can install it and use it
 
from setuptools import setup, find_packages

def get_requirments(file_path: str) -> list:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines() # read all the lines from the file
        requirements = [req.replace("\n", "") for req in requirements] # remove the new line character from each line
    
    # if -e . is present in the requirements, then remove it
    if "-e ." in requirements:
        requirements.remove("-e .")
    
    return requirements





setup(
    name = "ml_project",
    version= "0.0.1",
    author = "Umair Abid",
    author_email= "umairabid205@gmail.com",
    packages= find_packages(),
    install_requires= get_requirments('requirments.txt') #    # This function will read the requirements.txt file and return a list of packages
)

# After this, how it will be able to find that how many packages are there in the directory, I'll create a folder named "src"
# inside src folder I'll create a file named __init__.py
# This file will be empty, but it will tell Python that this is a package