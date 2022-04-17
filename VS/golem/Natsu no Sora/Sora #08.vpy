import vapoursynth as vs
import lvsfunc as lvf
import havsfunc as haf
import muvsfunc as mvs
from debandshit import dumb3kdb
from awsmfunc import bbmod
from vardefunc import nnedi3_upscale
from vsutil import depth
from kagefunc import retinex_edgemask
from adptvgrnMod import adptvgrnMod
from fine_dehalo import fine_dehalo

core = vs.core

src = lvf.src(r"AVS & YAPs\Sora #08 YAP.mp4", force_lsmas=True)
src = depth(src, 16)
src = lvf.deinterlace.fix_telecined_fades(src, tff=False)

nnscale = scale = nnedi3_upscale(src)

down_ssim = lvf.scale.ssim_downsample(nnscale, 864,480)
down_ssim = core.resize.Point(down_ssim, format=vs.YUV420P16, matrix_in_s="170m", matrix_s="170m")
down_dpid = core.dpid.Dpid(nnscale, 864, 480)
down_merge = core.std.Merge(down_ssim, down_dpid, .4)

post_scale = core.std.Crop(down_merge, 6,4,2,0)
post_scale = bbmod(post_scale, 2, 0, 0, 0)

cdenoise = haf.KNLMeansCL(post_scale, 2, 2, 4, 1)
cmerge = mvs.MergeChroma(post_scale, cdenoise, .8)

#magic = lvf.deblock.vsdpir(cmerge, mode='deblock', strength=15, tiles=4, cuda=True, matrix=6)

denoise = haf.SMDegrain(cmerge, 2, 50, 50, RefineMotion=True, contrasharp=True, prefilter=3)
denoise_mask =lvf.mask.detail_mask(denoise, 0, 1)
denoise_merge = core.std.MaskedMerge(denoise, cmerge, denoise_mask)

aa = lvf.aa.nneedi3_clamp(denoise_merge)

halo_inner = fine_dehalo(aa, rx=1.2, ry=1.1, darkstr=0, thmi=15, thma=50)
halo_outer = fine_dehalo(halo_inner, rx=2, ry=1.8, darkstr=0, thmi=25)

prefilter = haf.SMDegrain(halo_outer, tr=3, RefineMotion=True, thSAD=150, prefilter=3)
deband_mask = retinex_edgemask(prefilter)
diff = core.std.MakeDiff(halo_outer, prefilter)

deband = dumb3kdb(prefilter, radius=20, threshold=64, grain=0, use_neo=True)
deband = core.std.MergeDiff(deband, diff)
deband = core.std.MaskedMerge(deband, halo_outer, deband_mask)

grain = adptvgrnMod(deband, strength=0.23, size=1, luma_scaling=1, sharp=65, grain_chroma=False, static=True)

post_scale.set_output(2)
cmerge.set_output(3)
halo_outer.set_output(4)
prefilter.set_output(5)
deband.set_output(6)
grain.set_output(8)

kutsuna_1 = grain[6593:7299]
unknown_1 = grain[14209:14777]
Kutsuna_2 = grain[21566:23976]

final = depth(grain, 10)
final.set_output()