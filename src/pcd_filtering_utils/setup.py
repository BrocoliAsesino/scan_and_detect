from setuptools import find_packages, setup

package_name = "pcd_filtering_utils"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=[]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="alejandro.gonzalez@local.eurecat.org",
    maintainer_email="alejandro.gonzalez@eurecat.org",
    description="TODO: Package description",
    license="TODO: License declaration",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "plc_filters = test_scripts.plc_filters:main",
            "pcd_filtering_utils_server = ros2_server.server:main",
            "pcd_filtering_utils_test_client = ros2_server.test_client:main",
        ],
    },
)
