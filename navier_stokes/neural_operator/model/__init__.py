
from .ffno import Model as FFNO
from .cnext import Model as CNext
from .factformer import Model as FactFormer
from .percnn import Model as PeRCNN
from .dpot import Model as DPOT
from .ssno import Model as SSNO
from .sino import Model as SINO
from .ssnoe import Model as SSNOE
from .ssnoreal import Model as SSNOreal
from .dns import Model as DNS


model_dict = {
    'SSNO': SSNO,
    'SINO': SINO,
    'DNS': DNS,
    'SSNOE': SSNOE,
    'SSNOreal': SSNOreal,
    'FFNO': FFNO,
    'CNext': CNext,
    'FactFormer': FactFormer,
    'DPOT': DPOT,
    'PeRCNN': PeRCNN,
    # ...
}
