
from .ffno import Model as FFNO
from .cnext import Model as CNext
from .factformer import Model as FactFormer
from .percnn import Model as PeRCNN
from .dpot import Model as DPOT
from .ssno import Model as SSNO



model_dict = {
    'SSNO': SSNO,
    'FFNO': FFNO,
    'CNext': CNext,
    'FactFormer': FactFormer,
    'DPOT': DPOT,
    'PeRCNN': PeRCNN,
    # ...
}
