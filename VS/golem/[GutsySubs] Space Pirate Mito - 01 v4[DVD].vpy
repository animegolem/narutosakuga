import vapoursynth as vs
import lvsfunc as lvf
import debandshit as dbs
import havsfunc as haf
import muvsfunc as mvf
from awsmfunc import bbmod
from vsutil import depth, iterate
from adptvgrnMod import adptvgrnMod

core = vs.core

def dotnl(clip: vs.VideoNode) -> vs.VideoNode:
    tnlmeans = haf.KNLMeansCL(clip, h=1.5, a=3)
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
    grain = adptvgrnMod(deband_merge, strength=0.32, size=1.30, luma_scaling=1, sharp=95, lo=19, hi=200)
    return grain

src = core.d2v.Source(input=r"D:\1. Series Encodes\Space Pirate Mito\Disc 1\1.d2v", rff=False)

#too many dots 
lut = haf.LUTDeCrawl(src)
filter = lvf.rfs(src, lut, [(212, 459), (11509,11714)]) 

bits = depth(filter, 16)

prematch = haf.QTGMC(bits, FPSDivisor=2, SourceMatch=3, Lossless=2, NoiseProcess=0)
match = core.tivtc.TFM(bits, slow=2, clip2=prematch)
decimate = core.tivtc.TDecimate(match, mode=1)
vfrout = mvf.VFRSplice([
    decimate[:88], #
    match[110:202], #
    decimate[162:1729], #
    match[1579:1730], #
    decimate[1384:6654], #
    match[8318:8634], #
    decimate[6907:7354], #
    match[9192:9250], #
    decimate[7400:7942], #
    match[9928:9999], #
    decimate[7999:8106], #
    match[10133:10576], #
    decimate[8461:8547], #
    match[10684:10792], #
    decimate[8634:19302], #
    match[24128:24302], #
    decimate[19442:], #
], r"D:\timecodes.txt")
frate = core.std.AssumeFPS(vfrout, fpsnum=24000,fpsden=1001)
vin = haf.Vinverse(frate, amnt=30)

def bigcrop(clip: vs.VideoNode) -> vs.VideoNode:
    crop1 = clip[:18094] #
    crop1 = core.std.Crop(crop1, 8,8,0,0)
    crop1 = bbmod(crop1, left=1, right=2, top=0, bottom=2)
    cropbump = clip[18094:18367] #
    cropbump = core.std.Crop(cropbump, 8,8,0,4)
    cropbump = bbmod(cropbump, left=6, right=2, top=0, bottom=0)
    cropbump = core.std.AddBorders(cropbump, 0,0,2,2)
    crop3 = clip[18367:32528]
    crop3 = core.std.Crop(crop3, 8,8,0,0)
    crop3 = bbmod(crop3, left=1, right=2, top=0, bottom=2)
    cropED = clip[32528:34688] #
    cropED = core.std.Crop(cropED, 6,10,0,4)
    cropED= core.std.AddBorders(cropED, 0,0,2,2)
    cropED = bbmod(cropED, left=5, right=2, top=0, bottom=0)
    crop4 = clip[34688:] #
    crop4 = core.std.Crop(crop4, 8,8,0,0)
    crop4 = bbmod(crop4, left=0, right=0, top=0, bottom=0)
    crop_merge = core.std.Splice([crop1, cropbump, crop3, cropED, crop4], mismatch=1)
    scale = core.resize.Bicubic(crop_merge, 704,528)
    return scale
crop = bigcrop(vin)

dots = dotnl(crop)

c_denoise = haf.KNLMeansCL(dots, h=1, a=5)
chroma_merge = mvf.MergeChroma(dots, c_denoise, .7)

denoise = haf.SMDegrain(chroma_merge, tr=2, thSAD=50, RefineMotion=True, mfilter=chroma_merge, CClip=chroma_merge, contrasharp=True, prefilter=3)
denoise_mask = lvf.mask.detail_mask(denoise, 0, 1)
denoise_merge = core.std.MaskedMerge(denoise, chroma_merge, denoise_mask)

aa = based_clamp(denoise_merge)

grainyband = grainband(aa)

crop.set_output(3)

final = depth(grainyband, 10)
final.set_output()