#!/usr/bin/env python3
"""File-only assembly of an offline figure folder; no simulation or plotting."""
import argparse
import collections
import hashlib
import json
import pathlib
import re
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=pathlib.Path, required=True, help='Local documentation/reports folder')
    parser.add_argument('--destination', type=pathlib.Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    dest = args.destination.resolve()
    current = source / 'material_doublebang_openmp256_20260913'
    accepted = json.loads((current / 'acceptance.json').read_text())
    geometry = json.loads((current / 'geometry/geometry.json').read_text())
    assert accepted['accepted_count'] == 9 and len(geometry['events']) == 9
    dest.mkdir(parents=True, exist_ok=False)
    copied = []

    def copy(src, relative, category):
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        actual = sha(target)
        assert actual == sha(src)
        copied.append(dict(path=relative.as_posix(), category=category, source=str(src), bytes=target.stat().st_size, sha256=actual))

    def assets(folder, target, category):
        for src in sorted(folder.iterdir()):
            if src.is_file() and src.suffix.lower() in ['.png', '.pdf']:
                copy(src, pathlib.Path(target) / src.name, category)

    assets(current / 'geometry/figures', '01_geometry', 'Current 9-event geometry')
    for src in sorted((current / 'figures').glob('*.png')):
        category = '02_shower_profiles' if src.stem.endswith('_shower') else '03_radio_waveforms'
        copy(src, pathlib.Path(category) / src.name, category)
    assets(current / 'source_components/figures', '04_air_rock_components', 'Current air/rock source decomposition')
    assets(source / 'material_models_20260913/figures', '05_material_physics', 'Material-interface reference checks')
    references = [
        ('doublebang_radio_audit_20260912', '06_previous_reference/01_older_doublebang',
         '较早的三例 100 PeV ντ：旧 SiO₂ 配置 n=2，含旧事件几何和波形；不能当成本次 n=√5 的结果。'),
        ('interface_radio_figures_20260912', '06_previous_reference/02_real_rays_and_sp',
         '此前 1 GeV 电子、seed 67101 的真实光路重放、RadioPropa 对照和 s/p 分量图；不是本次 100 PeV 事件的射线。'),
        ('interface_radio_physics_20260912', '06_previous_reference/03_radio_validation',
         '此前射电传播、发射、收敛和 GPU 调度验证图：独立验证算例，保留其原参数。'),
    ]
    for report, target, description in references:
        assets(source / report / 'figures', target, description)
    for name in ['results.json', 'acceptance.json', 'campaign.json', 'progress.json']:
        p = current / name
        if p.exists():
            copy(p, pathlib.Path('data') / name, 'Current campaign metadata')
    copy(current / 'geometry/geometry.json', pathlib.Path('data/geometry.json'), 'Current geometry evidence')
    for src in sorted((current / 'geometry').glob('*.csv')):
        copy(src, pathlib.Path('data/terrain_sections') / src.name, 'DEM sections')
    for name in ['contributions.csv', 'contributions.json', 'provenance.json']:
        copy(current / 'source_components' / name, pathlib.Path('data') / name, 'Source decomposition data')
    for src in sorted(current.glob('openmp_*.json')):
        copy(src, pathlib.Path('data/event_summaries') / src.name, 'Current event summary')
    documentation = source.parent
    for name in ['material.md', 'materials.yaml']:
        copy(documentation / name, pathlib.Path('data/material_parameters') / name, 'User-supplied material specification')
    repo = documentation.parent
    for src in sorted((repo / 'configs/mountain/materials').glob('*.yaml')):
        copy(src, pathlib.Path('data/material_cards') / src.name, 'Native material cards')
    for name in ['geometry_figures_psr.py', 'analyze_psr.py', 'decompose_radio_psr.py', 'accept_psr.py']:
        copy(repo / 'validation/terrain/material_doublebang' / name, pathlib.Path('scripts') / name, 'PSR analysis source')

    # Preserve the useful completed-result notes with local, portable links.
    notes = dest / 'notes'
    notes.mkdir()
    text = (current / 'RADIO_COMPONENTS_CN.md').read_text()
    text = text.replace('source_components/figures/', '../04_air_rock_components/')
    text = text.replace('source_components/contributions.', '../data/contributions.')
    (notes / 'RADIO_COMPONENTS_CN.md').write_text(text)
    text = (current / 'ACCEPTANCE_CN.md').read_text().replace('(SLIDES_CN.md)', '(../README_CN.md)').replace('(acceptance.json)', '(../data/acceptance.json)')
    (notes / 'ACCEPTANCE_CN.md').write_text(text)
    captions = {
        'terrain_and_stations': '公共 DEM、入射位置、入射方向和 E01/E20/N10 三个接收站。',
        'material_lpm': '原生 PROPOSAL 与可移植 LPM 实现的核对；这是材料函数检查图。',
        'material_waveforms': '此前 10 GeV 电子材料验证算例的波形，不是本次高能事件。',
        'material_attenuation': '材料模型的电场衰减 exp(-l/L_E)，未单独包含几何扩散和界面透射。',
        'rock_air_contributions': '左图为山体源的独立电场平方积分占比，右图为相干总场峰值。',
        '01_real_mountain_path': '旧电子事件中的真实轨迹发射点、山体剖面及到 E01 的光路重放。',
        '02_real_refraction': '旧电子事件的山体内发射点、出射面与局部 Snell 折射几何。',
        '03_external_path_difference': '同端点的 beta5 与 RadioPropa 光路差异放大图。',
        '04_E01_waveform': '旧电子事件 E01 三分量波形；读取图中单位和到达时间。',
        '05_E01_coreas_zhs': '旧电子事件 CoREAS/ZHS 波形及复频谱差异。',
        '06_sp_geometry': 's 垂直入射面，p 位于入射面内；两者都垂直相应传播方向。',
        '07_sp_fresnel': 's/p 振幅透射与功率透射不同，注意分别读取 t 与 T。',
        '08_sp_actual_track': '旧单轨迹的 s/p 投影和局部传递矩阵，非整场簇射波形。',
    }

    def caption(path):
        if path.stem in captions:
            return captions[path.stem]
        if path.stem.endswith('_geometry_profile'):
            return '当前事件：上排 DEM 与真实 τ 轨迹，中排 τ 能量，下排 shower 能量沉积。'
        if path.stem.endswith('_shower'):
            return '当前同一种子的三介质能量沉积 profile；25 m 分箱，比较峰位和分布。'
        if path.stem.endswith('_components'):
            return '当前事件的空气源、山体源与相干总场；上排电场模，下排局部笛卡尔分量。'
        if path.stem.endswith('_waveforms'):
            return '查看完整接收窗及最强脉冲；实线 ZHS 与虚线 CoREAS，比较各面板刻度。'
        if path.stem.endswith('_topology'):
            return '旧版本事件的几何和 τ 能量演化；事件末态与当前版本可能不同。'
        return '较早验证图；横纵轴与图例均保留原文，所属算例见本组版本说明。'

    pngs = [p for p in copied if p['path'].endswith('.png')]
    grouped = collections.defaultdict(list)
    for record in pngs:
        grouped[pathlib.Path(record['path']).parent.as_posix()].append(record)
    index = ['# 全部图件索引', '', '[返回读图说明](README_CN.md)', '',
             '每行是一张独立图；若有 PDF，同时提供矢量版。页面截图和重复的报告渲染页未作为独立图计数。', '']
    for directory, records in grouped.items():
        index += ['## ' + directory, '', '| 图件 | 怎么看 | PDF |', '|---|---|---|']
        for record in records:
            path = pathlib.Path(record['path'])
            pdf = path.with_suffix('.pdf')
            pdf_link = '[PDF](%s)' % pdf.as_posix() if (dest / pdf).exists() else '—'
            index.append('| [%s](%s) | %s | %s |' % (path.stem, path.as_posix(), caption(path), pdf_link))
        index.append('')
    (dest / 'FIGURE_INDEX_CN.md').write_text('\n'.join(index))
    refdoc = ['# 之前的几何、光路、s/p 与射电验证图', '', '[返回主说明](README_CN.md)', '',
              '这些图保留原有输入与结果；本次九例的结果请看主文件夹 01–04。', '']
    for report, directory, description in references:
        refdoc += ['## ' + directory.split('/')[-1], '', description, '']
        for record in grouped[directory]:
            path = pathlib.Path(record['path'])
            refdoc += ['### ' + path.stem, '', '![](%s)' % path.as_posix(), '', caption(path), '']
    (dest / 'REFERENCE_CN.md').write_text('\n'.join(refdoc))
    short = lambda tag: 'SiO₂' if 'silica' in tag else 'CaCO₃' if 'limestone' in tag else '花岗岩混合物'
    modes = {'muonic': 'μ 子道', 'electronic': '电子道', 'hadronic': '强子道', 'air': '大气', 'rock': '山体'}
    current_pngs = sum(not p['path'].startswith('06_') for p in pngs)
    text = ['# beta5：九组不同介质 double-bang 模拟图集', '',
        '先看这份说明即可。所有图片标注为英文，下面用中文说明读法；同目录的相对链接可在 D 盘直接打开。', '',
        '**九组模拟均已完成并验收通过。** 100 PeV ντ，seed 158、946、3605；OpenMP 256 个物理核，LPM 开启，同时计算 CoREAS 与 ZHS。', '',
        '本文件夹共 **%d 张独立 PNG 图**，保留已有的 PDF 版本。其中 01–05 为当前几何、结果及材料接口参考图共 %d 张；06 为此前图件的参考附录。完整目录见 [全部图件索引](FIGURE_INDEX_CN.md)。' % (len(pngs), current_pngs), '',
        '| 山体模型 | 密度 kg/m³ | 折射率 | 电场衰减长度 m |', '|---|---:|---:|---:|',
        '| 二氧化硅 SiO₂ | 2650 | √5 | 100 |', '| 方解石／石灰岩 CaCO₃ | 2710 | √6 | 14.4765 |',
        '| 花岗岩混合物：H、C、O、Na、Mg、Al、Si、K、Ca、Fe | 2729 | √5 | 86.8589 |', '',
        '## 1. 先看共同的山体几何', '', '![](01_geometry/terrain_and_stations.png)', '',
        '底色是原始 DEM 高度，红色标记是入射位置，紫色三角形是 E01/E20/N10 接收站。箭头指向入射方向；虚线是入射轴与剖面位置指引。高度采用 ENU，相对原点；原点海拔约 2802 m。', '',
        '每个事件另有一张三联图：**上排看山体、CC 顶点及 τ 衰变位置，中排看 τ 能量，下排看 shower 能量沉积**。三排使用同一个“从注入点沿南向的距离”坐标。轨迹来自实际记录，剖面是固定 East 的投影；这里的粒子轨迹不是射电光路。', '',
        '## 2. 比较同一种子在三种介质中的 shower profile', '']
    for seed in [158, 946, 3605]:
        text += ['### seed %d' % seed, '', '![](02_shower_profiles/seed%d_shower.png)' % seed, '',
                 '不同颜色对应不同介质。纵轴为每公里的加权能量沉积，分箱 25 m；比较峰位、宽度和积分，注意这不是粒子数 profile。完整接收事件的随机发展可能不同，不能只按同种子推断同一 τ 衰变位置。', '']
    text += ['## 3. 九个事件：几何和射电一一对应', '',
             '| 介质 | seed | 实际 τ 衰变 | 几何、τ 能量和 profile | 三站射电波形 |', '|---|---:|---|---|---|']
    for row in accepted['accepted']:
        tag = row['tag']
        mode = '；'.join(modes[p['channel']] + '／' + modes[p['medium']] for p in row['decays'])
        text.append('| %s | %d | %s | [查看](01_geometry/%s_geometry_profile.png) | [查看](03_radio_waveforms/%s_waveforms.png) |' % (short(tag), row['seed'], mode, tag, tag))
    text += ['', '射电图上排是完整接收窗，下排放大最强脉冲；实线 ZHS、虚线 CoREAS。均采用 50–100 MHz 理想带通，未归一化、未平移时间。每站显示其最强脉冲处最强的笛卡尔分量；比较强弱要读纵轴倍率。电场尚未转换成天线电压。', '',
        '下面以 SiO₂／seed 946 为例：先在上排找信号时段，再在下排确认两算法是否重合。', '',
        '![](03_radio_waveforms/openmp_silica_SiO2_seed946_waveforms.png)', '',
        '## 4. 山体源与大气源分别贡献多少', '',
        '![](04_air_rock_components/rock_air_contributions.png)', '',
        '左图是山体源占比，大气源占比为 100% 减去该值；右图是相干总电场峰值，单位 nV/m，色标为对数。占比采用 Q=∫|E|²dt，以 Q山体+Q大气 为分母，干涉项单列。高占比不等于强信号。', '',
        '### 时间分开的例子：SiO₂／seed 158／E20', '',
        '![](04_air_rock_components/openmp_silica_SiO2_seed158_E20_components.png)', '',
        '蓝色大气源主峰约 22.984 μs，橙色山体源主峰约 41.160 μs，相隔 18.176 μs。黑色虚线是相干总场。上排画电场模，下排画同一个笛卡尔分量，纵轴分别取合适刻度。', '',
        '### 主峰重叠的例子：SiO₂／seed 3605／E20', '',
        '![](04_air_rock_components/openmp_silica_SiO2_seed3605_E20_components.png)', '',
        '山体源与大气源主峰只隔 3.906 ns，即一个采样点；主峰处不能靠简单时间窗完整拆开。山体 Q 占 46.88%，大气占 53.12%，干涉项占总 Q 约 −2.48%。', '',
        '[更多分量图和解释](notes/RADIO_COMPONENTS_CN.md) · [全部 27 个事件—站点的数值](data/contributions.csv)。分组依据辐射产生位置，既不是光路长度分组，也不等同于第一／第二个 bang。实际可探测性还需要天线响应和噪声。', '',
        '## 5. 材料模型的补充图', '',
        '| 图 | 怎么看 |', '|---|---|',
        '| [LPM 对照](05_material_physics/material_lpm.png) | 原生 PROPOSAL 与可移植实现是否重合。 |',
        '| [材料验证波形](05_material_physics/material_waveforms.png) | 此前 10 GeV 电子验证算例；不与本批 100 PeV 幅度直接比较。 |',
        '| [衰减尺度](05_material_physics/material_attenuation.png) | 路径每增加一个 L_E，电场因吸收降至约 37%。 |', '',
        '## 6. 之前的几何、真实光路和 s/p 图', '',
        '[打开参考图集](REFERENCE_CN.md)。这里完整保留旧三例的几何／波形、真实光路与 RadioPropa 比较、s/p 图，以及较早的射电物理验证图。每组都标明输入版本和用途。', '',
        '当前九组采用新材料接口；旧三例 SiO₂ 使用 n=2，旧真实光路图来自 1 GeV 电子事例。当前九组的 CC、τ 位置与射电结果，以 01–04 为准。', '',
        '## 参数与验收', '',
        '[验收说明](notes/ACCEPTANCE_CN.md) · [机器可读验收结果](data/acceptance.json) · [几何及投影信息](data/geometry.json)。参数源文件及可选材料卡保存在 `data/material_parameters/` 和 `data/material_cards/`。', '',
        '本批沿用原截断、thinning 和 100 μs 输运窗。数值与调度检查通过，尚未完成这些近似的收敛扫描。几何补图和所有数值绘图均在 PSR 进行；本地仅整理文件。每个复制文件的来源和散列见 `FILE_MANIFEST.json`。', '']
    (dest / 'README_CN.md').write_text('\n'.join(text))
    # Portable Markdown links and byte-for-byte copied assets are file checks,
    # not local simulation tests. Do not invoke numerical dependencies here.
    missing = []
    for md in dest.rglob('*.md'):
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', md.read_text()):
            if '://' in link or link.startswith('#'):
                continue
            target = (md.parent / link.split('#', 1)[0]).resolve()
            if not target.exists():
                missing.append(dict(document=str(md.relative_to(dest)), link=link))
    assert not missing, missing
    output_manifest = dict(destination=str(dest), independent_png_figures=len(pngs), copied_files=len(copied),
        all_copied_hashes_match=True, markdown_relative_links_valid=True, files=copied,
        generated_documents=[dict(path=p.relative_to(dest).as_posix(), sha256=sha(p)) for p in sorted(dest.rglob('*.md')) if p.name not in ['material.md']])
    (dest / 'FILE_MANIFEST.json').write_text(json.dumps(output_manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(destination=str(dest), png_figures=len(pngs), pdf_figures=sum(p['path'].endswith('.pdf') for p in copied),
                         copied_files=len(copied), copied_bytes=sum(p['bytes'] for p in copied), links_valid=True), ensure_ascii=False))


if __name__ == '__main__':
    main()
