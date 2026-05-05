from dataclasses import dataclass

@dataclass
class Config:

    # Native settings #
    RR_refrac_sensitivity: float = 0.5 #in sec
    deriv_tightness: int = 6 #MIN 3
    calib_window: float = 10   #in sec
    srate_hz: float = 10 
    process: bool = True
    filter_type: str = "b" 

    # Preprocessing Low Pass #
    min_cutoff: float = 1.0
    beta: float = 0.0
    d_cutoff: float = 1.0

    # Preprocessing Band Pass #
    low: float = 0.05                 # low cutoff, Hz
    high: float = 0.7                 # low cutoff, Hz
    order: int = 2
 
