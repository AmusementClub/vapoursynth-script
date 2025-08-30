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

assert rip.format.color_family in [vs.YUV, vs.GRAY], "only support YUV or Gray input"

def force8bit(clip):
    if clip.format.bits_per_sample == 8:
        return clip
    return clip.resize.Spline64(format=clip.format.replace(bits_per_sample=8).id, dither_type='none')
src = force8bit(src)
rip = force8bit(rip)

# Get the clips in same dimention.
if src.width != rip.width or src.height != rip.height or src.format.name != rip.format.name:
    src = src.resize.Bicubic(rip.width, rip.height, format=rip.format)

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
