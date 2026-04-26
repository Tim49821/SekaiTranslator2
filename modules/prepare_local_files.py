from pathlib import Path
from typing import Union, List
import os.path as osp
import os
import traceback

from . import INPAINTERS, TEXTDETECTORS, OCR, TRANSLATORS
from .base import BaseModule, LOGGER
import utils.shared as shared
from utils.download_util import download_and_check_files


def _required_file_exists(save_dir: str, required_file: Union[str, List[str]]) -> bool:
    if isinstance(required_file, (list, tuple, set)):
        return any(_required_file_exists(save_dir, candidate) for candidate in required_file)

    save_path = Path(save_dir)
    if any(token in required_file for token in ['*', '?', '[']):
        return any(save_path.glob(required_file))
    return (save_path / required_file).exists()


def _hf_snapshot_is_ready(save_dir: str, required_files: List[Union[str, List[str]]]) -> bool:
    if not osp.isdir(save_dir):
        return False
    return all(_required_file_exists(save_dir, required_file) for required_file in required_files)


def download_and_check_hf_model_files(module_class: BaseModule):
    repo_id = getattr(module_class, 'hf_model_repo_id', None)
    save_dir = getattr(module_class, 'hf_model_save_dir', None)
    if repo_id is None or save_dir is None:
        return True

    required_files = getattr(module_class, 'hf_model_required_files', ['config.json'])
    if _hf_snapshot_is_ready(save_dir, required_files):
        return True

    if os.environ.get('BALLOONTRANS_SKIP_HF_MODEL_DOWNLOAD', '').strip().lower() in {'1', 'true', 'yes'}:
        LOGGER.warning(f'Skipping Hugging Face model download for {module_class}: BALLOONTRANS_SKIP_HF_MODEL_DOWNLOAD is set.')
        return False

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        LOGGER.error(f'Failed to import huggingface_hub while preparing {repo_id}: {e}')
        return False

    try:
        LOGGER.info(f'downloading Hugging Face model {repo_id} to {save_dir} ...')
        snapshot_download(
            repo_id=repo_id,
            local_dir=save_dir,
            local_dir_use_symlinks=False,
            token=os.environ.get('HF_TOKEN') or None,
            allow_patterns=getattr(module_class, 'hf_model_allow_patterns', None),
            ignore_patterns=getattr(module_class, 'hf_model_ignore_patterns', None),
        )
    except Exception:
        LOGGER.error(
            f'Failed downloading Hugging Face model {repo_id}. '
            f'If this is a gated model, log in with huggingface-cli or set HF_TOKEN.'
        )
        LOGGER.error(f'Please manually save the model snapshot to {save_dir}.')
        LOGGER.error(traceback.format_exc())
        return False

    if not _hf_snapshot_is_ready(save_dir, required_files):
        LOGGER.error(f'Hugging Face model {repo_id} was downloaded, but required files are still missing in {save_dir}.')
        return False

    return True


def should_prepare_hf_model(module_class: BaseModule) -> bool:
    if getattr(module_class, 'hf_model_repo_id', None) is None:
        return False
    if getattr(module_class, 'hf_model_download_on_prepare', False):
        return True
    return os.environ.get('BALLOONTRANS_DOWNLOAD_HF_MODEL_ON_PREPARE', '').strip().lower() in {'1', 'true', 'yes'}


def download_and_check_module_files(module_class_list: List[BaseModule] = None):
    if module_class_list is None:
        module_class_list = []
        for registered in [INPAINTERS, TEXTDETECTORS, OCR, TRANSLATORS]:
            for module_key in registered.module_dict.keys():
                module_class_list.append(registered.get(module_key))

    for module_class in module_class_list:
        if should_prepare_hf_model(module_class):
            download_and_check_hf_model_files(module_class)

        if module_class.download_file_on_load or module_class.download_file_list is None:
            continue
        for download_kwargs in module_class.download_file_list:
            all_successful = download_and_check_files(**download_kwargs)
            if all_successful:
                continue
            LOGGER.error(f'Please save these files manually to sepcified path and restart the application, otherwise {module_class} will be unavailable.')

def prepare_pkuseg():
    try:
        import pkuseg
    except:
        import spacy_pkuseg as pkuseg

    flist = [
        {
            'url': 'https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/postag.zip',
            'files': ['features.pkl', 'weights.npz'],
            'sha256_pre_calculated': ['17d734c186a0f6e76d15f4990e766a00eed5f72bea099575df23677435ee749d', '2bbd53b366be82a1becedb4d29f76296b36ad7560b6a8c85d54054900336d59a'],
            'archived_files': 'postag.zip',
            'save_dir': 'data/models/pkuseg/postag'
        },
        {
            'url': 'https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip',
            'files': ['features.msgpack', 'weights.npz'],
            'sha256_pre_calculated': ['fd4322482a7018b9bce9216173ae9d2848efe6d310b468bbb4383fb55c874a18', '5ada075eb25a854f71d6e6fa4e7d55e7be0ae049255b1f8f19d05c13b1b68c9e'],
            'archived_files': 'spacy_ontonotes.zip',
            'save_dir': 'data/models/pkuseg/spacy_ontonotes'
        },
    ]
    for files_download_kwargs in flist:
        download_and_check_files(**files_download_kwargs)

    PKUSEG_HOME = osp.join(shared.PROGRAM_PATH, 'data/models/pkuseg')
    pkuseg.config.pkuseg_home = PKUSEG_HOME

    # there must be data/models/pkuseg/postag.zip and data/models/pkuseg/spacy_ontonotes.zip
    # otherwise the dumb package download these models again becuz its dumb checking
    p = osp.join(PKUSEG_HOME, 'postag.zip')
    if not osp.exists(p):
        os.makedirs(p)

    p = osp.join(PKUSEG_HOME, 'spacy_ontonotes.zip')
    if not osp.exists(p):
        os.makedirs(p)


def prepare_local_files_forall():

    # download files required by detect, ocr, inpaint and translators
    download_and_check_module_files()

    prepare_pkuseg()

    if shared.CACHE_UPDATED:
        shared.dump_cache()
