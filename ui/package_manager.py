import os

from modules.package_import_names import PACKAGE_IMPORT_NAMES
from utils.config import pcfg
from utils.network_mirrors import installer_env_with_pypi_mirror
from utils.py_package_manager import PyPackageManager


def create_package_manager() -> PyPackageManager:
    pmcfg = pcfg.package_manager
    return PyPackageManager(
        backend=pmcfg.installer_backend,
        extra_args=pmcfg.extra_install_args,
        package_import_names=PACKAGE_IMPORT_NAMES,
        env=installer_env_with_pypi_mirror(os.environ.copy(), pcfg.mirrors.pypi),
    )
