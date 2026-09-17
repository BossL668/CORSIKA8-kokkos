#!/usr/bin/env python3
"""File transfer only: mirror completed PSR figures/status; no local analysis."""
import argparse
import datetime
import io
import json
import pathlib
import shlex
import subprocess
import tarfile
import time

REMOTE = '/data/yhlu/CorsikaData/corsika_validation_results/beta5_SiO2_high_energy_aircuts_thin1e-5_20260914'
COLLECT = r'''
import pathlib,sys,tarfile
root=pathlib.Path(sys.argv[1])
paths=[root/'campaign.json',root/'progress.json']+list((root/'preflight').glob('*.json'))
paths+=list((root/'report').rglob('*'))
paths+=list((root/'runs').glob('*/status.json'))
paths+=list((root/'runs').glob('*/material_checks.json'))
paths+=list((root/'runs').glob('*/doublebang_classification.json'))
paths+=list((root/'timing').glob('*.json'))
paths+=list((root/'timing/runs').glob('*/status.json'))
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for path in sorted(set(paths)):
  if path.is_file() and path.suffix in ['.png','.pdf','.md','.json','.csv']:
   archive.add(path,arcname=str(path.relative_to(root)),recursive=False)
'''


def sync(destination, control):
    command = ['ssh', '-S', control, '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
               'psrpku2025_PKU', 'python3 -c ' + shlex.quote(COLLECT) + ' ' + shlex.quote(REMOTE)]
    result = subprocess.run(command, capture_output=True, check=True, timeout=300)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:') as archive:
        for item in archive:
            if not item.isfile():
                continue
            name = pathlib.PurePosixPath(item.name)
            if name.is_absolute() or '..' in name.parts:
                raise RuntimeError('Unexpected archive path')
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = archive.extractfile(item).read()
            if not target.exists() or target.read_bytes() != data:
                temporary = target.with_suffix(target.suffix + '.download')
                temporary.write_bytes(data)
                temporary.replace(target)
    progress = json.loads((destination / 'progress.json').read_text())
    stamp = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    lines = ['# SiO₂ 更高能量 double bang：计算状态', '',
             '状态同步时间：' + stamp + '。', '',
             '**%s；已完成 %d 个，候选上限 %d 个。**' % (progress['state'], len(progress['completed']), progress['total']), '',
             '当前步骤：`' + progress.get('current', 'finished') + '`。', '',
             '300 PeV、1 EeV 各预备最多 30 个不同种子，以每档得到至少 3 个实际 double bang 为目标。保留未通过拓扑筛选的结果。保持原 SiO₂、DEM、入射位置和方向、三个接收站；OpenMP 256 核，常驻队列，CoREAS/ZHS 同算。输运窗 500 μs，接收窗 2048 μs，采样率 256 MHz。', '',
             '大气默认粒子截断：电磁 0.5 MeV，强子/μ/τ 0.3 GeV。按用户要求 emthin=1e-5；最大权重按大气自动公式计算，300 PeV 为 1500、1 EeV 为 5000。旧 emthin=0.01 试跑已停止并废弃。', '',
             '全部模拟、编译、检查和绘图在 PSR；此文件夹只接收结果文件。当前尚未完成的事件不会被当作结果展示。', '']
    if (destination / 'report/README_CN.md').exists():
        lines += ['[打开结果图与读图说明](report/README_CN.md)', '']
    else:
        lines += ['第一组完成并通过检查后，结果图和读图说明会由服务器生成并同步到此处。', '']
    if (destination / 'report/TIMING_CN.md').exists():
        lines += ['[低能计时实测与高能耗时估算](report/TIMING_CN.md)', '']
        if progress['state'].startswith('timing_pilot'):
            lines += ['本阶段先做 10 TeV、100 TeV、1 PeV 递增计时；300 PeV / 1 EeV 正式队列暂未启动。', '']
    if (destination / 'report/LIVE_TIMING_CN.md').exists():
        lines += ['[计时曲线与最近一次吞吐率快照](report/LIVE_TIMING_CN.md)', '']
    if progress.get('error'):
        lines += ['停止原因：' + progress['error'], '']
    if progress.get('qualified'):
        lines += ['已通过谱系筛选：' + '；'.join('%g PeV：%d 个' % (int(k)/1e6,len(v)) for k,v in progress['qualified'].items()) + '。', '']
    if (destination / 'preflight/result.json').exists():
        oracle = json.loads((destination / 'preflight/result.json').read_text())
        lines += ['LPM 预检查已通过：SiO₂ 原生/可移植实现，最高 1 EeV，%d 点，最大相对差异 %.3g。' % (oracle['points'], oracle['maximum_relative_error']), '']
    lines += ['服务器目录：`' + REMOTE + '`。', '',
              '完整输入：[campaign.json](campaign.json)。实时服务器状态：[progress.json](progress.json)。', '']
    (destination / 'README_CN.md').write_text('\n'.join(lines))
    print(stamp, progress['state'], len(progress['completed']), '/', progress['total'], flush=True)
    return progress['state']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', required=True, type=pathlib.Path)
    parser.add_argument('--control', required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    deadline = time.monotonic() + 30 * 24 * 3600
    while True:
        try:
            state = sync(args.destination, args.control)
            if state in ['complete', 'failed', 'superseded', 'candidate_limit_reached', 'timing_pilot_complete', 'timing_pilot_incomplete'] or not args.watch:
                return
        except (OSError, subprocess.SubprocessError, tarfile.TarError) as error:
            print(datetime.datetime.now().isoformat(), 'SYNC ERROR', str(error), flush=True)
            if not args.watch:
                raise
        if time.monotonic() >= deadline:
            raise RuntimeError('30-day file synchronization limit; remote campaign is independent')
        time.sleep(120)


if __name__ == '__main__':
    main()
