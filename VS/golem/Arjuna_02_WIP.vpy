import vapoursynth as vs
import lvsfunc as lvf
import debandshit as dbs
import havsfunc as haf
import muvsfunc as mvf
import EoEfunc as eoe
from ccd import ccd
from vsutil import depth
from awsmfunc import bbmod
from adptvgrnMod import adptvgrnMod
from vardefunc import nnedi3_upscale

core = vs.core

###Loading and FPS###
src = core.d2v.Source(input=r"D:\arjuna\1\2.d2v", rff=False)
src = depth(src, 16)
src = lvf.deinterlace.TIVTC_VFR(src, tfm_in=r"D:\arjuna\1\matches2.txt", tdec_in=r"D:\arjuna\1\metrics2.txt", timecodes_out=r"D:\arjuna\1\timecodes2.txt")
src = core.std.AssumeFPS(src, fpsnum=24000,fpsden=1001)
src = core.std.Crop(src, 6,4,0,0)
src = bbmod(src, left=2, right=2, top=0, bottom=0)

###Major Filters###
dpir = lvf.deblock.vsdpir(src, strength=15, mode='deblock', cuda=True, matrix=6)
#dpir2 = lvf.deblock.vsdpir(src, strength=7, mode='deblock', cuda=True, matrix=6)
chroma_merge = mvf.MergeChroma(src, dpir, 1)
denoise = eoe.denoise.CMDegrain(chroma_merge, thSAD=100)
edge = haf.EdgeCleaner(denoise)

###Tweaks, Banding, and Grain###
###AA###
aa_light = lvf.aa.transpose_aa(edge, eedi3=False)
aa_heavy = lvf.aa.based_aa(edge, shader_file=r"D:\2. VapourSynth Scripts\.plugins\FSRCNNX_x2_56-16-4-1.glsl")
aa_merge = lvf.aa.clamp_aa(edge, aa_light, aa_heavy, 1)
###halo###
dehalo = haf.FineDehalo(aa_merge, rx=2, ry=.4, darkstr=.2, brightstr=1, thlimi=25)
###Deband###
deband = dbs.dumb3kdb(dehalo, 16, 30, 0, 4, True)                       
deband_mask = lvf.mask.detail_mask(dehalo, 0, 1)
deband_merge = core.std.MaskedMerge(deband, dehalo, deband_mask)
######
grain = adptvgrnMod(deband_merge,strength=0.25, size=1.0, luma_scaling=8, sharp=80, static=False, lo=19, hi=200)

###Booru Scaling###
#scale = nnedi3_upscale()
#dscale = core.resize.Bicubic()

###Booru cuts###
Trim = core.std.Trim(grain, 4758, 5020)

###Diagnostics###

stack_start = lvf.comparison.stack_planes(src)
stack_end = lvf.comparison.stack_planes(grain)
comparator = lvf.comparison.stack_compare(src, grain)

###Outputs###
src.set_output(1)
aa_merge.set_output(2)
dehalo.set_output(3)
comparator.set_output(4)
stack_end.set_output(5)

###Final###
final = depth(grain, 10)
final.set_output()
