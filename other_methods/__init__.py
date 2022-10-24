from .clever import CLEVER
from .fast_lip import FastLip, FastLip2
from .lip_lp import LipLP
# from .lip_sdp import LipSDP
from .naive_methods import NaiveUB, RandomLB
from .seq_lip import SeqLip
from .other_methods import OtherResult
from .z_lip import ZLip
from .lipopt_file import LipOpt

OTHER_METHODS = [CLEVER, FastLip, FastLip2, LipLP, NaiveUB, RandomLB, SeqLip, ZLip]#, LipSDP]
LOCAL_METHODS = [CLEVER, FastLip, FastLip2, LipLP, RandomLB, ZLip]
GLOBAL_METHODS = [NaiveUB, SeqLip]#, LipSDP