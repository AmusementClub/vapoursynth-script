# vapoursynth-script

## MIRVsFunc

### Dependencies

- [lexpr](https://github.com/AkarinVS/vapoursynth-plugin)
- [fmtconv](https://github.com/EleonoreMizo/fmtconv)
- [bm3dcpu / bm3dcuda / bm3dcuda_rtc](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA)

### Usage

#### RgTools

`RgTools(clip, planes=[0,1,2]).RemoveGrain(mode, radius=1)` = `rgvs.RemoveGrain(clip, mode)`

`RgTools(clip, planes=[0,1,2]).Repair(repclip, mode)` = `rgvs.Repair(clip, repclip, mode)`