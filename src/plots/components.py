"""Scientific figure components; data are read from recomputed tables."""
from pathlib import Path
import hashlib
import textwrap
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[2]/'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np
import pandas as pd
from ..paths import ROOT, OUTPUT
PROJECT=ROOT
WIDTH=5.5
BLUE='#276B91'
RUST='#AE543B'
INK='#202020'
GRAY='#9BA0A4'
FAINT='#D9DDE0'
PAPER='#F3F5F6'
TASKS=['verbalizer','tg','conflict']
TITLES=['Source description','Resource transfer','Source choice']
UNITS=[r'$\Delta S$ (log ratio)',r'$\Delta\mathbb{E}[a]$ (points)',r'$\Delta D$ (log odds)']
INPUTS,CHECKS={},{}
OUT=OUTPUT/'figures'
def table(name):
    return pd.read_csv(OUTPUT/'tables'/f'{name}.csv')


def ci(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260925)
    means = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return (values.mean(), *np.quantile(means, [0.025, 0.975]))


def axis(ax, grid=None):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.65)
    ax.tick_params(length=3, width=0.65, pad=3)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color='#E3E5E6', lw=0.55)


def panel(fig, x, y, letter, title):
    fig.text(x, y, letter, weight='bold', fontsize=11, va='top')
    fig.text(x + 0.036, y - 0.002, title, fontsize=9.3, weight='bold', va='top')


def xerr(ax, x, lo, hi, y, color, marker='o', ms=4.2):
    ax.errorbar(x, y, xerr=[[max(0, x - lo)], [max(0, hi - x)]], color=color, fmt=marker, ms=ms, capsize=2.2, capthick=0.7, elinewidth=0.95, zorder=5)


def box(ax, values, position, color, width=0.38, horizontal=False, white=False):
    return ax.boxplot([np.asarray(values)], positions=[position], widths=width, vert=not horizontal, patch_artist=True, whis=1.5, showmeans=False, showfliers=True, manage_ticks=False, boxprops={'facecolor': 'white' if white else color, 'edgecolor': color, 'alpha': 0.85, 'linewidth': 0.8}, medianprops={'color': INK, 'linewidth': 1.1}, whiskerprops={'color': color, 'linewidth': 0.8}, capprops={'color': color, 'linewidth': 0.8}, flierprops={'marker': 'o', 'markerfacecolor': 'none', 'markeredgecolor': color, 'markersize': 2.8, 'markeredgewidth': 0.7})


def export(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overflow = []
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        box = artist.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1 + 1 or (box.y1 > fig.bbox.y1 + 1):
            overflow.append(artist.get_text())
    CHECKS[name] = {'canvas_inches': fig.get_size_inches().tolist(), 'overflow': overflow}
    assert not overflow, (name, overflow)
    fig.savefig(OUT / f'{name}.pdf', facecolor='white', metadata={'Creator': 'Anonymous scientific figures'})
    fig.savefig(OUT / f'{name}.svg', facecolor='white')
    fig.savefig(OUT / f'{name}.png', dpi=300, facecolor='white')
    plt.close(fig)


def behavior():
    summary, runs = (table('behavior_summary'), table('behavior_run_metrics'))
    fig = plt.figure(figsize=(WIDTH, 4.0))
    panel(fig, 0.26, 0.98, 'a', 'Resource allocation')
    panel(fig, 0.7, 0.98, 'b', 'Information choice')
    a = fig.add_axes([0.27, 0.18, 0.34, 0.68])
    b = fig.add_axes([0.71, 0.18, 0.26, 0.68])
    fig.legend(handles=[Rectangle((0, 0), 1, 1, facecolor=BLUE, label='75% accurate'), Rectangle((0, 0), 1, 1, facecolor=RUST, label='25% accurate')], loc='upper left', bbox_to_anchor=(0.25, 0.922), ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.0, fontsize=8.4)
    for y, r in enumerate(summary.itertuples()):
        sub = runs[runs.model == r.model]
        if r.model in ['qwen27b', 'gemma31b']:
            for ax in [a, b]:
                ax.axhspan(y - 0.46, y + 0.46, color=PAPER, zorder=0)
        for col, color, offset in [('tg_high', BLUE, -0.17), ('tg_low', RUST, 0.17)]:
            box(a, sub[col], y + offset, color, width=0.25, horizontal=True)
        p = 100 * r.conflict_rate
        b.barh(y, p, height=0.55, color=BLUE, zorder=2)
        b.barh(y, 100 - p, left=p, height=0.55, color=RUST, zorder=2)
        b.text(p / 2, y, f'{p:.1f}%', color='white', ha='center', va='center', fontsize=8.2, zorder=4)
    a.set_yticks(range(8), summary.label, fontsize=8.6)
    for label, model in zip(a.get_yticklabels(), summary.model):
        label.set_weight('bold' if model == 'qwen27b' else 'normal')
    a.tick_params(axis='y', length=0, pad=8)
    a.set_xlim(-0.5, 10.5)
    a.set_xticks([0, 2, 4, 6, 8, 10])
    a.set_xlabel('Transfer (points)')
    b.set_xlim(0, 100)
    b.set_xticks([0, 50, 100])
    b.set_xlabel('Share of conflict choices (%)')
    b.set_yticks([])
    b.axvline(50, color='#444444', ls=(0, (2, 3)), lw=0.65, zorder=1)
    for ax in [a, b]:
        ax.set_ylim(7.65, -0.65)
        axis(ax)
        ax.spines['left'].set_visible(False)
    a.tick_params(axis='y', length=0)
    fig.text(0.27, 0.035, 'Boxes: history-level class means. Bars: 240 conflict choices per model.', fontsize=8.0, color='#555555')
    export(fig, 'f2_behavior')


def descriptions():
    base = table('baseline_all_readouts')
    data = base[base.readout == 'verbalizer']
    lex = table('publication_word_effects_clustered').drop_duplicates(['group', 'word'])
    stats = table('verbalizer_summary').iloc[0]
    values = [data[data.label == label].baseline.to_numpy() for label in [0, 1]]
    assert all((len(v) == 20 for v in values))
    fig = plt.figure(figsize=(WIDTH, 4.15))
    fig.text(0.035, 0.975, 'After the same TAG history, query one partner:', fontsize=9.2, va='top')
    fig.text(0.035, 0.906, 'Participant B6J has been ____.', family='Courier New', fontsize=10, va='top')
    fig.text(0.035, 0.843, 'Output one English adjective only.', fontsize=8.5, color='#555555', va='top')
    panel(fig, 0.035, 0.75, 'a', 'Frozen descriptive vocabulary')
    for group, y, color, label in [('H', 0.654, BLUE, '$V_H$: 75%-history association (14 words)'), ('L', 0.355, RUST, '$V_L$: 25%-history association (11 words)')]:
        words = sorted(lex[lex.group == group].word)
        fig.text(0.035, y, label, color=color, fontsize=8.5, weight='bold', va='top')
        fig.text(0.035, y - 0.061, textwrap.fill(', '.join(words), 43), fontsize=8.6, linespacing=1.42, va='top', color=INK)
    panel(fig, 0.61, 0.75, 'b', 'Reliability separation')
    ax = fig.add_axes([0.65, 0.27, 0.31, 0.375])
    violins = ax.violinplot(values, positions=[0, 1], widths=0.72, showmeans=False, showextrema=False, bw_method='scott', points=100)
    for part, color in zip(violins['bodies'], [RUST, BLUE]):
        part.set_facecolor(color)
        part.set_edgecolor(color)
        part.set_alpha(0.28)
        part.set_linewidth(0.8)
    for label, color in [(0, RUST), (1, BLUE)]:
        box(ax, values[label], label, color, width=0.12, white=True)
    ax.set_xticks([0, 1], ['25% history', '75% history'])
    ax.set_xlim(-0.52, 1.52)
    ax.set_ylim(data.baseline.min() - 0.8, data.baseline.max() + 0.8)
    ax.set_ylabel('Description score $S$')
    axis(ax)
    ax.axhline(0, color=GRAY, ls=(0, (3, 3)), lw=0.7)
    fig.text(0.65, 0.135, f'High - low: {stats.high_low_gap:.2f}\n95% CI [{stats.gap_low:.2f}, {stats.gap_high:.2f}]', fontsize=8.6, linespacing=1.4)
    fig.text(0.035, 0.047, '20 partner queries per class, nested in 10 histories. Inner boxes: median and quartiles.', fontsize=8.0, color='#555555')
    export(fig, 'f3_description')


def steering():
    data = table('intervention_records')
    data = data[data.task_type == 'steer']
    stats = table('steering_run_clustered')
    fig = plt.figure(figsize=(WIDTH, 3.55))
    fig.text(0.05, 0.975, 'One description-derived vector, applied at the queried partner ID', fontsize=9.4, weight='bold', va='top')
    fig.text(0.05, 0.895, "$h'_{43,t}=h_{43,t}+\\alpha s\\,\\hat v/\\sqrt{m}$", fontsize=11, va='top')
    fig.text(0.56, 0.895, 'Participant B6J', family='Courier New', fontsize=9.5, va='top', bbox={'facecolor': PAPER, 'edgecolor': 'none', 'pad': 3})
    fig.text(0.88, 0.891, '4 tokens', fontsize=8.3, color='#555555', va='top')
    for i, task in enumerate(TASKS):
        x = 0.095 + i * 0.315
        panel(fig, x - 0.04, 0.765, chr(97 + i), TITLES[i])
        ax = fig.add_axes([x, 0.235, 0.235, 0.405])
        s = stats[stats.readout == task].sort_values('alpha')
        ax.plot(s.alpha, s.mean_delta, color=INK, lw=1.1, zorder=2)
        for alpha, color, marker in [(-2, RUST, 's'), (0, INK, 'o'), (2, BLUE, 'o')]:
            r = s[s.alpha == alpha].iloc[0]
            ax.errorbar(alpha, r.mean_delta, yerr=[[r.mean_delta - r.ci_low], [r.ci_high - r.mean_delta]], fmt=marker, color=color, ms=5.8, elinewidth=1.35, capsize=3, zorder=4)
        ax.axhline(0, color=GRAY, lw=0.7, ls=(0, (3, 3)))
        ax.set_xlim(-2.65, 2.65)
        ax.set_xticks([-2, 0, 2], ['−2', '0', '+2'])
        extent = max(abs(s.ci_low.min()), abs(s.ci_high.max())) * 1.18
        ax.set_ylim(-extent, extent)
        axis(ax)
        ax.set_ylabel(UNITS[i], labelpad=4, fontsize=9)
    fig.text(0.5, 0.115, 'Steering strength $\\alpha$', ha='center', fontsize=9.2)
    fig.text(0.05, 0.035, '10 evaluation histories; mean changes and 95% history-bootstrap intervals.', fontsize=8.0, color='#555555')
    export(fig, 'f4_steering')


def patching():
    data = table('intervention_records')
    data = data[data.task_type == 'patch']
    example = data[(data.pair_id == 0) & (data.readout == 'tg') & (data.source_label == 1)].iloc[0]
    fig = plt.figure(figsize=(WIDTH, 4.6))
    panel(fig, 0.04, 0.985, 'a', "Replace the same partner's query state across histories")
    top = fig.add_axes([0.04, 0.69, 0.92, 0.21])
    top.set_axis_off()
    top.text(0.0, 0.96, 'Donor history: 12/16 accurate', color=BLUE, fontsize=8.7, weight='bold', va='top')
    top.text(0.6, 0.96, 'Recipient: 4/16 accurate', color=RUST, fontsize=8.7, weight='bold', va='top')
    for x, color in [(0, BLUE), (0.6, RUST)]:
        top.text(x, 0.54, 'Participant B6J', family='Courier New', fontsize=9, va='center', bbox={'facecolor': 'white', 'edgecolor': color, 'pad': 3, 'linewidth': 0.8})
        top.text(x, 0.1, 'ID-span residuals, layer 43', fontsize=8, va='center', color='#555555')
    top.annotate('', xy=(0.59, 0.54), xytext=(0.34, 0.54), arrowprops={'arrowstyle': '->', 'lw': 1.25, 'color': INK})
    top.text(0.46, 0.74, 'patch', ha='center', fontsize=8.7)
    fig.text(0.04, 0.615, f'Example pair 01: expected transfer {example.baseline:.2f} → {example.changed:.2f} points.', fontsize=9)
    for i, task in enumerate(TASKS):
        x = 0.1 + i * 0.315
        panel(fig, x - 0.045, 0.54, chr(98 + i), TITLES[i])
        ax = fig.add_axes([x, 0.18, 0.235, 0.285])
        sub = data[data.readout == task]
        for source, xp, color in [(1, 0, BLUE), (0, 1, RUST)]:
            rows = sub[sub.source_label == source].sort_values('pair_id')
            box(ax, rows.delta, xp, color, width=0.44)
        extent = max(abs(sub.delta.min()), abs(sub.delta.max())) * 1.18
        ax.set_ylim(-extent, extent)
        ax.set_xlim(-0.45, 1.45)
        ax.axhline(0, color=GRAY, ls=(0, (3, 3)), lw=0.7)
        ax.set_xticks([0, 1], ['75→25', '25→75'])
        ax.set_ylabel(UNITS[i], fontsize=9, labelpad=3)
        axis(ax)
    fig.text(0.5, 0.075, 'Donor → recipient report accuracy (%)', ha='center', fontsize=9)
    fig.text(0.04, 0.018, '10 matched pairs sharing 9 histories; boxes summarize the two directional distributions.', fontsize=8.0, color='#555555')
    export(fig, 'f5_patching')


def random_controls():
    control_path = OUTPUT / 'tables/all_tasks.csv'
    INPUTS[str(control_path.relative_to(PROJECT))] = hashlib.sha256(control_path.read_bytes()).hexdigest()
    control = pd.read_csv(control_path)
    rand = control[control.condition == 'random_id_span_steer']
    data = table('intervention_records')
    keys = ['run_id', 'advisor_id', 'readout', 'alpha']
    data = data[(data.task_type == 'steer') & (data.alpha != 0)].merge(rand[keys].drop_duplicates(), on=keys, validate='one_to_one')
    fig = plt.figure(figsize=(WIDTH, 3.45))
    for i, task in enumerate(['verbalizer', 'tg']):
        x = 0.095 + i * 0.49
        panel(fig, x - 0.045, 0.96, chr(97 + i), TITLES[i])
        ax = fig.add_axes([x, 0.28, 0.36, 0.56])
        groups = [data[data.readout == task].groupby('run_id').oriented.mean()]
        groups += [rand[(rand.readout == task) & (rand.random_index == j)].groupby('run_id').oriented.mean() for j in sorted(rand.random_index.unique())]
        for j, values in enumerate(groups):
            color = BLUE if j == 0 else GRAY
            box(ax, values, j, color, width=0.52)
        ax.axhline(0, color=INK, lw=0.75, ls=(0, (3, 3)))
        ax.set_xticks(range(6), ['Reliability', 'R1', 'R2', 'R3', 'R4', 'R5'], rotation=45, ha='right')
        ax.set_ylabel(['Oriented $\\Delta S$', 'Oriented transfer change (points)'][i], fontsize=9)
        ax.set_xlim(-0.55, 5.5)
        axis(ax)
    fig.text(0.05, 0.025, 'Same 10 histories, one partner each; identical layer, ID span and perturbation norm.', fontsize=8, color='#555555')
    export(fig, 's1_random_controls')


def words():
    data = table('publication_word_effects_clustered')
    cmap = LinearSegmentedColormap.from_list('probability_change', [RUST, '#FAFAFA', BLUE])
    fig = plt.figure(figsize=(WIDTH, 4.65))
    norm = TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=0.4)
    for group, x, title in [('H', 0.32, '$V_H$: high-history association'), ('L', 0.8, '$V_L$: low-history association')]:
        sub = data[data.group == group].pivot(index='word', columns='alpha', values='mean').sort_index()
        panel(fig, x - 0.28, 0.975, 'a' if group == 'H' else 'b', title)
        height = 0.72 * len(sub) / 14
        ax = fig.add_axes([x, 0.88 - height, 0.145, height])
        ax.imshow(sub[[-2, 2]].values, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_yticks(range(len(sub)), sub.index, fontsize=8.6)
        ax.set_xticks([0, 1], ['$-2$', '$+2$'])
        ax.xaxis.tick_top()
        ax.tick_params(length=0, pad=4)
        for row in range(len(sub)):
            for col, alpha in enumerate([-2, 2]):
                value = sub.iloc[row][alpha]
                ax.text(col, row, f'{value:+.2f}', ha='center', va='center', fontsize=7.5, color='white' if abs(value) > 0.25 else INK)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_yticks(np.arange(-0.5, len(sub), 1), minor=True)
        ax.set_xticks([-0.5, 0.5, 1.5], minor=True)
        ax.grid(which='minor', color='white', lw=1)
        ax.tick_params(which='minor', length=0)
    colorax = fig.add_axes([0.32, 0.075, 0.4, 0.023])
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(mappable, cax=colorax, orientation='horizontal', ticks=[-0.4, -0.2, 0, 0.2, 0.4])
    cb.outline.set_visible(False)
    fig.text(0.52, 0.014, 'Mean change in complete-answer log probability', ha='center', fontsize=8.7)
    export(fig, 's2_word_changes')


def behavior_distribution():
    data = table('behavior_run_metrics')
    summary = table('behavior_summary')
    fig = plt.figure(figsize=(WIDTH, 5.2))
    panel(fig, 0.035, 0.97, 'a', 'Within-history transfer differences')
    a = fig.add_axes([0.29, 0.585, 0.66, 0.3])
    hist = []
    for j, r in enumerate(summary.itertuples()):
        sub = data[data.model == r.model].sort_values('run_id')
        box(a, sub.tg_gap, j, BLUE, width=0.53, horizontal=True)
        counts = np.bincount(np.rint(8 * sub.conflict_high_rate).astype(int), minlength=9)
        assert counts.sum() == 30
        hist.append(counts)
    a.set_yticks(range(8), summary.label, fontsize=8)
    a.set_ylim(7.6, -0.6)
    a.set_xlabel('TG high - low (points)')
    a.axvline(0, color=GRAY, ls=(0, (3, 3)), lw=0.8)
    axis(a)
    a.tick_params(axis='y', length=0)
    a.set_xlim(-6, 10)
    panel(fig, 0.035, 0.46, 'b', 'Distribution of conflict choices across histories')
    b = fig.add_axes([0.29, 0.09, 0.66, 0.29])
    cmap = LinearSegmentedColormap.from_list('history_count', ['#FFFFFF', BLUE])
    b.imshow(hist, cmap=cmap, vmin=0, vmax=30, aspect='auto', interpolation='nearest')
    b.set_xticks(range(9))
    b.set_yticks(range(8), summary.label, fontsize=8)
    b.set_xlabel('High-reliability choices per history (out of 8)')
    b.tick_params(length=0)
    for y, counts in enumerate(hist):
        for x, count in enumerate(counts):
            if count:
                b.text(x, y, str(count), ha='center', va='center', fontsize=8, color='white' if count > 16 else INK)
    for sp in b.spines.values():
        sp.set_visible(False)
    export(fig, 's3_behavior_histories')
