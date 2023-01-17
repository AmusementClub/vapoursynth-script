import vapoursynth as vs
import types
import sys
from vapoursynth import core
from importlib.machinery import SourceFileLoader


# Load the source clip by loading another vs script. 

sourceScript = r"OKE:SOURCE_SCRIPT"
videoFile = r"OKE:VIDEO_FILE"
loader = SourceFileLoader('source', sourceScript)
mod = types.ModuleType(loader.name)
OKE:VSPIPE_ARGS
loader.exec_module(mod)

src = vs.get_output(index = 1)

# Load the ripped file.
rip = core.lsmas.LWLibavSource(videoFile)

if rip.format.subsampling_h == 1 and rip.format.subsampling_w == 1:
    format = vs.YUV420P8
elif rip.format.subsampling_h == 0 and rip.format.subsampling_w == 0:
    format = vs.YUV444P8
else:
    assert False, "Only Support YUV420 and YUV444 format!"

if not (src.format.bits_per_sample == 8): src = src.resize.Spline64(format=format, dither_type='none')
if not (rip.format.bits_per_sample == 8): rip = rip.resize.Spline64(format=format, dither_type='none')

# Get the clips in same dimention.
if src.width != rip.width or src.height != rip.height:
    src = src.resize.Bicubic(rip.width, rip.height)

src_planes = [ src.std.ShufflePlanes(i, vs.GRAY) for i in range(src.format.num_planes) ]
rip_planes = [ rip.std.ShufflePlanes(i, vs.GRAY) for i in range(rip.format.num_planes) ]

# runtime function to print psnr value to console.
def callback(n, f):
    print("RPCOUT:", n, ' '.join([ '%f' % p.props.PlanePSNR for p in f ]), flush=True, file=sys.stderr)
    return rip

# Compare. Run it using 'vspipe "00009.m2ts_rpc.vpy" NUL'. 
cmp_planes = [ core.complane.PSNR(a, b) for (a, b) in zip(rip_planes, src_planes) ]
cmp = core.std.FrameEval(rip, callback, prop_src=cmp_planes)
cmp.set_output()
