
from .ffno import Model as FFNO
from .cnext import Model as CNext
from .factformer import Model as FactFormer
from .percnn import Model as PeRCNN
from .dpot import Model as DPOT
from .ssno import Model as SSNO
from .sino import Model as SINO
from .sino2 import Model as SINO2


model_dict = {
    'SSNO': SSNO,
    'SINO': SINO,
    'SINO2': SINO2,
    'FFNO': FFNO,
    'CNext': CNext,
    'FactFormer': FactFormer,
    'DPOT': DPOT,
    'PeRCNN': PeRCNN,
    # ...
}
