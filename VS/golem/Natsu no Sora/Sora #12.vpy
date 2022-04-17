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

src = lvf.src(r"AVS & YAPs\Sora #12 YAP.mp4", force_lsmas=True)
src = depth(src, 16)
ep = src[0:29772]
creds = src[29772:33917]
creds = core.std.AssumeFPS(creds, fpsnum=30000, fpsden=1001) 
creds = lvf.deinterlace.ivtc_credits(creds, 3, tff=True) #3
creds = haf.Vinverse(creds, amnt=30)

src2 = ep + creds
src2 = core.std.AssumeFPS(src2, fpsnum=24000, fpsden=1001) 
fades = lvf.deinterlace.fix_telecined_fades(src2, tff=False)

nnscale = scale = nnedi3_upscale(fades)

down_ssim = lvf.scale.ssim_downsample(nnscale, 864,480)
src = core.std.SetFrameProp(src, prop="_Matrix", intval=6)
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

kutsuna1 = grain[6737:7283] + grain[7373:7607] + grain[12322:12673]
medly = grain[17555:17745] + grain[17840:18640] + grain[19298:19819] + grain[19848:20182] + grain[20225:20703] + grain[21056:21397] 
kutsuna2 = grain[28799:29425]
dontforget = grain[32850:32984]
photozoom = grain[13248:13748]


src.set_output(2)

final = depth(grain, 10)
final.set_output()


