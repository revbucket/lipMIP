from .clever import CLEVER
from .fast_lip import FastLip 
from .lip_lp import LipLP
from .lip_sdp import LipSDP 
from .naive_methods import NaiveUB, RandomLB
from .seq_lip import SeqLip 
from .other_methods import OtherResult

OTHER_METHODS = [CLEVER, FastLip, LipLP, LipSDP, NaiveUB, RandomLB, SeqLip]
