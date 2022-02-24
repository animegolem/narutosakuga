import vapoursynth as vs
import lvsfunc as lvf
import havsfunc as haf
import debandshit as dbs
from vsutil import depth
from adptvgrnMod import adptvgrnMod

core = vs.core

def based_clamp(clip: vs.VideoNode) -> vs.VideoNode:
    aa_light = lvf.aa.transpose_aa(clip, eedi3=False)
    aa_heavy = lvf.aa.based_aa(clip, shader_file=r"D:\2. VapourSynth Scripts\.plugins\FSRCNNX_x2_56-16-4-1.glsl", gamma=30, alpha=.25)
    aa_merge = lvf.aa.clamp_aa(clip, aa_light, aa_heavy, 1)
    return aa_merge
def dumbband(clip: vs.VideoNode) -> vs.VideoNode:
    deband = dbs.dumb3kdb(clip, 16, 30, 0, 4, True)                       
    deband_mask = lvf.mask.detail_mask(clip, 0, 1)
    deband_merge = core.std.MaskedMerge(deband, clip, deband_mask)
    grain = adptvgrnMod(deband_merge,strength=0.25, size=1.0, luma_scaling=8, sharp=80, static=False, lo=19, hi=200)
    return grain

src = lvf.src(r"C:\Users\golem\Desktop\M2F2\M4.MOV[0].avi", force_lsmas=True)
src = depth(src, 16)

dpir = lvf.deblock.vsdpir(src, strength=20, mode='deblock', cuda=True, matrix=1)
smd = haf.SMDegrain(dpir, tr=2, thSAD=100, CClip=dpir, mfilter=dpir, contrasharp=True, prefilter=3)
edge = haf.EdgeCleaner(smd, strength=12)
aa = based_clamp(edge)
deband = dumbband(aa)

src.set_output(1)
dpir.set_output(2)
final = depth(deband, 8)
final.set_output()
