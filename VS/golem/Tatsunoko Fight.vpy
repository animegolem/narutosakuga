import vapoursynth as vs
import lvsfunc as lvf
from vsutil import depth
from vsTAAmbk import TAAmbk
from adptvgrnMod import adptvgrnMod

core = vs.core

#loading and fps
src = lvf.src(r"C:\Users\****\Desktop\STR\MOV1\VOL_OP.STR[0].avi", force_lsmas=True)
src = core.std.Levels(src, min_in=0, max_in=255, min_out=16, max_out=235, planes=0)
src = core.std.Levels(src, min_in=0, max_in=255, min_out=16, max_out=240, planes=[1,2])
src = depth(src, 16)

#Filters
dpir = lvf.deblock.vsdpir(src, strength=50, mode='deblock', cuda=True, matrix=1)
dpir2 = lvf.deblock.vsdpir(src, strength=40, mode='deblock', cuda=True, matrix=1)
dpir3 = lvf.deblock.vsdpir(src, strength=60, mode='deblock', cuda=True, matrix=1)

aa = TAAmbk(dpir, aatype='Nnedi3')
deband = core.neo_f3kdb.Deband(aa, range=15, y=64, cb=64, cr=64, grainy=15, grainc=15, dynamic_grain=True, sample_mode=2)
grain = adptvgrnMod(deband,strength=0.25, size=1.0, luma_scaling=8, sharp=80, static=False, lo=19, hi=200)

trim1 = core.std.Trim(grain, 211, 567)
trim2 = core.std.Trim(grain, 758,1176)

#outputs
src.set_output(1)
dpir.set_output(2)
dpir2.set_output(3)
dpir3.set_output(4)
grain.set_output(5)

final = depth(trim2, 8)
final.set_output()


