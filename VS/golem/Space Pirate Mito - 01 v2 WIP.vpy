import vapoursynth as vs
import lvsfunc as lvf
import debandshit as dbs
import havsfunc as haf
import muvsfunc as mvf
from awsmfunc import bbmod
from vsutil import depth, iterate
from adptvgrnMod import adptvgrnMod

core = vs.core
def bigcrop(clip: vs.VideoNode) -> vs.VideoNode:
    crop1 = clip[0:17382]
    crop1 = core.std.Crop(crop1, 8,8,0,0)
    crop1 = bbmod(crop1, left=1, right=2, top=0, bottom=2)
    cropbump = clip[17382:17655]
    cropbump = core.std.Crop(cropbump, 8,8,0,4)
    cropbump = bbmod(cropbump, left=6, right=2, top=0, bottom=0)
    cropbump = core.std.AddBorders(cropbump, 0,0,2,2)
    crop3 = clip[17655:31784]
    crop3 = core.std.Crop(crop3, 8,8,0,0)
    crop3 = bbmod(crop3, left=1, right=2, top=0, bottom=2)
    cropED = clip[31784:33939]
    cropED = core.std.Crop(cropED, 6,10,0,4)
    cropED= core.std.AddBorders(cropED, 0,0,2,2)
    cropED = bbmod(cropED, left=5, right=2, top=0, bottom=0)
    crop4 = clip[33939:34418]
    crop4 = core.std.Crop(crop4, 8,8,0,0)
    crop4 = bbmod(crop4, left=0, right=0, top=0, bottom=0)
    crop_merge = core.std.Splice([crop1, cropbump, crop3, cropED, crop4], mismatch=1)
    scale = core.resize.Bicubic(crop_merge, 704,528)
    return scale
def dotnl(clip: vs.VideoNode) -> vs.VideoNode:
    tnlmeans = tnlmeans = haf.KNLMeansCL(crop, h=1.5, a=3)
    dpir = lvf.deblock.vsdpir(tnlmeans, strength=35, mode='deblock', cuda=True, matrix=6)
    smd = haf.SMDegrain(dpir, tr=3, thSAD=200, RefineMotion=True, mfilter=dpir, CClip=dpir, contrasharp=True, prefilter=3)
    line_mask = lvf.mask.detail_mask(smd, 0, 1)
    mask_THICK = iterate(line_mask, core.std.Inflate, 10)
    SMDpir_merge = core.std.MaskedMerge(clip, smd, mask_THICK)
    return SMDpir_merge
def based_clamp(clip: vs.VideoNode) -> vs.VideoNode:
    aa_light = lvf.aa.transpose_aa(clip, eedi3=False)
    aa_heavy = lvf.aa.based_aa(clip, shader_file=r"D:\2. VapourSynth Scripts\.plugins\FSRCNNX_x2_56-16-4-1.glsl", gamma=30, alpha=.25)
    aa_merge = lvf.aa.clamp_aa(clip, aa_light, aa_heavy, 1)
    return aa_merge
def grainband(clip: vs.VideoNode) -> vs.VideoNode:
    deband = dbs.dumb3kdb(clip, 16, 64, 0, 4, True)                       
    deband_mask = lvf.mask.detail_mask(clip, 0, 1)
    deband_merge = core.std.MaskedMerge(deband, clip, deband_mask)
    grain = adptvgrnMod(deband_merge,strength=0.25, size=1.4, luma_scaling=8, sharp=80, static=False, lo=19, hi=200)
    return grain

src = core.d2v.Source(input=r"D:\1. Series Encodes\Space Pirate Mito\Disc 1\1.d2v", rff=False)

#too many dots 
lut = haf.LUTDeCrawl(src)
filter = lvf.rfs(src, lut, [(212, 459)])

ivtc = depth(filter, 16)
ivtc = core.tivtc.TFM(ivtc)
ivtc = core.tivtc.TDecimate(ivtc)
ivtc = core.std.AssumeFPS(ivtc, fpsnum=24000,fpsden=1001)
ivtc = haf.Vinverse(ivtc, amnt=30)
ivtc = lvf.deinterlace.fix_telecined_fades(ivtc)
crop = bigcrop(ivtc)

dots = dotnl(crop)
c_denoise = haf.KNLMeansCL(dots, h=1, a=5)
chroma_merge = mvf.MergeChroma(dots, c_denoise, 1)
denoise = haf.SMDegrain(chroma_merge, tr=2, thSAD=75, RefineMotion=True, mfilter=chroma_merge, CClip=chroma_merge, contrasharp=True, prefilter=3)
denoise_mask = lvf.mask.detail_mask(denoise, 0, 1)
denoise_merge = core.std.MaskedMerge(denoise, chroma_merge, denoise_mask)

aa = based_clamp(denoise_merge)
grainyband = grainband(aa)

crop.set_output(1)
final = depth(grainyband, 10)
final.set_output()
