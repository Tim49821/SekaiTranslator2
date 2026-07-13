from utils.io_utils import build_funcmap
from utils.fontformat import FontFormat
from utils.config import pcfg
from utils.textblock_mask import (
    canny_flood,
    canny_flood_natural,
    connected_canny_flood,
    existing_mask,
)


MASKSEG_METHOD_1 = 0
MASKSEG_METHOD_2 = 1
MASKSEG_EXISTING_MASK = 2
MASKSEG_METHOD_3 = 3

MASKSEG_METHODS = {
    MASKSEG_METHOD_1: canny_flood,
    MASKSEG_METHOD_2: connected_canny_flood,
    MASKSEG_EXISTING_MASK: existing_mask,
    MASKSEG_METHOD_3: canny_flood_natural,
}

# Build base function map
handle_ffmt_change = build_funcmap('ui.fontformat_commands', 
                                     list(FontFormat.params().keys()) + ['rel_font_size'], 
                                     'ffmt_change_', verbose=False)


def get_maskseg_method(method_id=None):
    if method_id is None:
        method_id = pcfg.drawpanel.rectool_method
    return MASKSEG_METHODS.get(method_id, canny_flood)
