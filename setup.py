from setuptools import find_packages,setup

REQUIREMENT_FILE_NAME='requirements.txt'

# "pip install -r requirements.txt" will automaticaly run the setup.py file
HYPHEN_E_DOT = '-e .'
def get_requirements():
    # read REQUIREMENT_FILE_NAME
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
    # replace \n with "" in list
    requirement_list = [name.replace('\n', "") for name in requirement_list]
    # if requirement_list contains "-e ." remote that
    if HYPHEN_E_DOT in requirement_list:
        requirement_list.remove(HYPHEN_E_DOT)
    # return requirement_list we made
    return requirement_list

setup(
    name="sensor",
    version="0.0.1",
    author="ineuron",
    author_email="hiteshwadhwani1403@gmai.com",
    packages = find_packages(),
    install_requires=get_requirements(),
)