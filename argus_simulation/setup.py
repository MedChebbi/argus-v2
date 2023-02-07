import os
from glob import glob
from setuptools import setup


package_name = 'argus_simulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, "worlds"), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, "models", "argus_v2"), glob('models/argus_v2/*')),
        (os.path.join('share', package_name, "models", "platform/meshes/"), glob('models/platform/meshes/*.dae')),
        (os.path.join('share', package_name, "models", "platform/materials/textures/"), glob('models/platform/materials/textures/*')),
        (os.path.join('share', package_name, "models", "platform/"), glob('models/platform/*.sdf')),
        (os.path.join('share', package_name, "models", "platform/"), glob('models/platform/*.config')),
        (os.path.join('share', package_name, "models", "robolympics_circuit/meshes/"), glob('models/robolympics_circuit/meshes/*.dae')),
        (os.path.join('share', package_name, "models", "robolympics_circuit/materials/textures/"), glob('models/robolympics_circuit/materials/textures/*')),
        (os.path.join('share', package_name, "models", "robolympics_circuit/"), glob('models/robolympics_circuit/*.sdf')),
        (os.path.join('share', package_name, "models", "robolympics_circuit/"), glob('models/robolympics_circuit/*.config')),
    ],

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohamed',
    maintainer_email='1.mohamed.chebbi@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
