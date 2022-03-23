import vapoursynth as vs
import lvsfunc as lvf
import havsfunc as haf
import debandshit as dbs
from vsutil import depth, get_y
from awsmfunc import bbmod
from vardefunc import fsrcnnx_upscale
from vardefunc.scale import nnedi3_upscale
from adptvgrnMod import adptvgrnMod

core = vs.core

src = lvf.src(r"C:\Users\golem\Desktop\Gad Guard NCED Yatta.mp4", force_lsmas=True)
src = depth(src, 16)

crop2 = core.std.Crop(src, 8,8,2,0)
bb = bbmod(crop2, 2, 0, 2, 2)

yscale = fsrcnnx_upscale(clip=get_y(bb), width=640, height=480, shader_file=r"D:\2. VapourSynth Scripts\.plugins\FSRCNNX_x2_56-16-4-1.glsl", profile="zastin")
cscale = core.resize.Bicubic(bb,width=640, height=480, filter_param_a=0, filter_param_b=1/2)
scale = core.std.ShufflePlanes(clips=[yscale, cscale], planes=[0, 1, 2], colorfamily=vs.YUV)

wherethemagichappens = lvf.deblock.vsdpir(scale, 10, mode='deblock', tiles=8, matrix=6, cuda=True)
aa_light = lvf.aa.taa(wherethemagichappens,lvf.aa.nnedi3(opencl=True))
aa_heavy = lvf.aa.transpose_aa(wherethemagichappens, eedi3=False)
aa_merge = lvf.aa.clamp_aa(wherethemagichappens, aa_light, aa_heavy, 1)
grain = adptvgrnMod(aa_merge, strength=0.2, size=1.4, luma_scaling=1, sharp=75, lo=19, hi=200)

src.set_output(2)

final = depth(grain, 8)
final.set_output()

#https://www.sakugabooru.com/post/show/183866