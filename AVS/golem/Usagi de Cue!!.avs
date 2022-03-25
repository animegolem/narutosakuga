d2vsource("C:\Users\golem\Desktop\VTS_01_1.d2v")

checkmate()

TFM()
Tdecimate(mode=1)

crop(8,2,-8,-0)
BicubicResize(640,480)

SMDegrain(2, 100, refinemotion=true, prefilter=6)
prefetch(4)

santiag(nns=4)

finedehalo(2.5,1.5, darkstr=0)
finedehalo(2.5,1.5, darkstr=0)
finedehalo(2.5,1.5, darkstr=0)
prefetch(4)

F3KDB_3(range=16, y=48, cb=44, cr=44, grainy=0, grainc=0, dynamic_grain=true)
AdaptiveGrain(AddGrain(var=.25, uvar=0.0, constant=False))

trim(3000,4000)