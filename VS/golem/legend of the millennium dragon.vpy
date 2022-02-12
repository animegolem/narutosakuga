import vapoursynth as vs
import lvsfunc as lvf
import debandshit as dbs
from vsutil import depth
from adptvgrnMod import adptvgrnMod

core = vs.core

src = lvf.src(r"Z:\Torrent Downloads\Legend.of.the.Millennium.Dragon.2011.Blu-Ray.1080p.AVC.DTS-HD.MA5.1-CHDBits\BDMV\STREAM\00800.m2ts", force_lsmas=True)
src = depth(src, 16)
src = core.std.Crop(src, 0,0,132,130)
src = core.resize.Bicubic(src, 1128, 480)

deband = dbs.dumb3kdb(src, 16, 30, 0, 4, True)                       
deband_mask = lvf.mask.detail_mask(src, 0, 1)
deband_merge = core.std.MaskedMerge(deband, src, deband_mask)
grain = adptvgrnMod(deband_merge,strength=0.25, size=1.0, luma_scaling=8, sharp=80, static=False, lo=19, hi=200)

cut1 = core.std.Trim(grain, 121342,121586)
cut2 = core.std.Trim(grain, 121587,121754)
cut3 = core.std.Trim(grain, 121755,121849)
booru = core.std.Trim(grain, 121850, 122680)

final = depth(booru, 8)
final.set_output()