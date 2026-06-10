from .ffno import Model as FFNO
from .percnn import Model as PeRCNN
from .dpot import Model as DPOT
from .cnext import Model as CNext
from .ssno import Model as SSNO



model_dict = {
    'SSNO': SSNO,
    'FFNO': FFNO,
    'CNext': CNext,
    'DPOT': DPOT,
    'PeRCNN': PeRCNN
}
