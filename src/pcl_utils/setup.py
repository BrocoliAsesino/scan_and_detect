from setuptools import find_packages, setup

package_name = 'pcl_utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test', 'resource']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'open3d',
    ],
    zip_safe=True,
    maintainer='alejandro.gonzalez@local.eurecat.org',
    maintainer_email='alejandro.gonzalez@eurecat.org',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
                'plc_filters = test_scripts.plc_filters:main',
        ],
    },
)
