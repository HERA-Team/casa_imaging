from setuptools import setup
import glob
import os
import sys
import json

def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths

data_files = package_files('casa_imaging', 'data') + package_files('casa_imaging', '../scripts')

setup_args = {
    'name': 'casa_imaging',
    'author': 'HERA Team',
    'url': 'https://github.com/HERA-Team/casa_imaging',
    'license': 'BSD',
    'description': 'Collection of scripts for calibration, imaging and data processing in CASA and Python.',
    'package_dir': {'casa_imaging': 'casa_imaging'},
    'packages': ['casa_imaging'],
    'include_package_data': True,
    'scripts': ['scripts/pbcorr.py', 'scripts/source2file.py', 'scripts/make_model_cube.py',
                'scripts/skynpz2calfits.py', 'scripts/source_extract.py',
                'scripts/find_sources.py'],
    'version': '0.1',
    'package_data': {'casa_imaging': data_files},
    'zip_safe': False,
}

if __name__ == '__main__':
    setup(*(), **setup_args)
