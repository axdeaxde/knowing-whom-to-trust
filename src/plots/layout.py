"""Final composite figure geometry."""
from . import components as b
from . import plot_data as replay
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
from ..paths import OUTPUT
OUT=OUTPUT/'figures'
LAYOUT={}


def capture(fun):
    figures = []
    original = b.export
    b.export = lambda fig, name: figures.append((fig, name))
    try:
        fun()
    finally:
        b.export = original
    assert len(figures) == 1
    fig, name = figures[0]
    return (fig, name)


def arrange(fig, name, height, axes_bottoms, text_anchors):
    old_height = fig.get_figheight()
    old_axes = [ax.get_position().bounds for ax in fig.axes]
    before_text = [text.get_text() for text in fig.texts]
    fig.set_size_inches(b.WIDTH, height)
    axes_audit = []
    for ax, (x, _, w, h), bottom in zip(fig.axes, old_axes, axes_bottoms):
        physical_height = h * old_height
        ax.set_position([x, bottom / height, w, physical_height / height])
        axes_audit.append({'original_height_inches': physical_height, 'new_height_inches': ax.get_position().height * height})
    for artist in fig.texts:
        x, y = artist.get_position()
        anchor = min(text_anchors, key=lambda old: abs(old - y))
        assert abs(anchor - y) < 0.006, (name, artist.get_text(), y)
        artist.set_position((x, (text_anchors[anchor] + (y - anchor) * old_height) / height))
    for index, artist in enumerate(fig.texts):
        if artist.get_text() in list('abcd'):
            artist.set_text(f'({artist.get_text()})')
            title = fig.texts[index + 1]
            x, y = title.get_position()
            title.set_position((x + 0.021, y))
    LAYOUT[name] = {'original_canvas_height': old_height, 'new_canvas_height': height, 'axes': axes_audit, 'original_figure_text': before_text}


def finish(fig, name, allowed_text_changes=None):
    after = [artist.get_text() for artist in fig.texts]
    original = LAYOUT[name]['original_figure_text']
    changes = []
    for previous, current in zip(original, after):
        if previous != current:
            assert current == f'({previous})' and previous in 'abcd' or (allowed_text_changes or {}).get(previous) == current
            changes.append({'old': previous, 'new': current})
    assert len(original) == len(after)
    LAYOUT[name]['figure_text_changes'] = changes
    assert all((np.isclose(row['original_height_inches'], row['new_height_inches']) for row in LAYOUT[name]['axes']))
    b.export(fig, name)


def violin(ax, values, x, color):
    values = np.asarray(values, dtype=float)
    if np.ptp(values) > 0:
        body = ax.violinplot(values, positions=[x], widths=0.72, showmeans=False, showextrema=False, bw_method='scott', points=100)['bodies'][0]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
        body.set_linewidth(0.8)
    b.box(ax, values, x, color, width=0.12, white=True)


def behavior():
    fig, name = capture(b.behavior)
    arrange(fig, name, 3.8, [0.6, 0.6], {0.98: 3.75, 0.035: 0.1})
    fig.legends[0].set_bbox_to_anchor((0.25, 3.57 / 3.8))
    finish(fig, name)


def descriptions():
    fig, name = capture(b.descriptions)
    arrange(fig, name, 3.3, [0.81], {0.975: 3.24, 0.906: 3.03, 0.843: 2.82, 0.75: 2.56, 0.654: 2.28, 0.593: 2.07, 0.355: 1.32, 0.294: 1.11, 0.135: 0.33, 0.047: 0.065})
    finish(fig, name)


def steering():
    fig, name = capture(b.steering)
    arrange(fig, name, 3.15, [0.805] * 3, {0.975: 3.09, 0.895: 2.82, 0.765: 2.45, 0.115: 0.35, 0.035: 0.08})
    stats = replay.fresh('dose_summary')
    bands = []
    for i, (ax, task) in enumerate(zip(fig.axes, b.TASKS)):
        ax.clear()
        sub = stats[stats.readout == task].sort_values('alpha')
        x = sub.alpha.to_numpy()
        ax.fill_between(x, sub.ci_low.to_numpy(), sub.ci_high.to_numpy(), color=b.BLUE, alpha=0.34, linewidth=0, zorder=1)
        ax.plot(x, sub.ci_low, color=b.BLUE, lw=0.55, alpha=0.65, zorder=2)
        ax.plot(x, sub.ci_high, color=b.BLUE, lw=0.55, alpha=0.65, zorder=2)
        ax.plot(x, sub.mean_delta, color=b.BLUE, lw=1.15, zorder=3)
        ax.axhline(0, color=b.GRAY, lw=0.7, ls=(0, (3, 3)))
        ax.set_xlim(-4.35, 4.35)
        ax.set_xticks(replay.ALPHAS, ['−4', '−2', '−1', '0', '1', '2', '4'], fontsize=8)
        extent = max(abs(sub.ci_low.min()), abs(sub.ci_high.max())) * 1.1
        ax.set_ylim(-extent, extent)
        if task == 'conflict':
            ax.set_yticks([-0.4, 0, 0.4])
        elif task == 'tg':
            ax.set_yticks([-0.5, 0, 0.5])
        b.axis(ax)
        ax.set_ylabel(b.UNITS[i], labelpad=4, fontsize=9)
        bands.extend(({'readout': task, 'alpha': int(row.alpha), 'ci_width': float(row.ci_high - row.ci_low), 'mean_delta': float(row.mean_delta), 'ci_low': float(row.ci_low), 'ci_high': float(row.ci_high)} for row in sub.itertuples()))
    pd.DataFrame(bands).to_csv(OUT / 'confidence_band_values.csv', index=False)
    finish(fig, name)


def patching():
    fig, name = capture(b.patching)
    arrange(fig, name, 3.85, [2.6, 0.68, 0.68, 0.68], {0.985: 3.8, 0.615: 2.365, 0.54: 2.16, 0.075: 0.3, 0.018: 0.075})
    data = b.table('intervention_records')
    data = data[data.task_type == 'patch']
    for i, (ax, task) in enumerate(zip(fig.axes[1:], b.TASKS)):
        ylim, xlim = (ax.get_ylim(), ax.get_xlim())
        ax.clear()
        sub = data[data.readout == task]
        for label, x, color in [(1, 0, b.BLUE), (0, 1, b.RUST)]:
            violin(ax, sub[sub.source_label == label].sort_values('pair_id').delta, x, color)
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.axhline(0, color=b.GRAY, ls=(0, (3, 3)), lw=0.7)
        ax.set_xticks([0, 1], ['75→25', '25→75'])
        ax.set_ylabel(b.UNITS[i], fontsize=9, labelpad=3)
        b.axis(ax)
    old = '10 matched pairs sharing 9 histories; boxes summarize the two directional distributions.'
    new = old.replace('boxes summarize', 'violins summarize')
    fig.texts[-1].set_text(new)
    finish(fig, name, {old: new})


def random_controls():
    fig, name = capture(b.random_controls)
    arrange(fig, name, 3.15, [0.85, 0.85], {0.96: 3.015, 0.025: 0.075})
    paired = replay.fresh('random_paired_rows')
    paired = paired[paired.magnitude == 4]
    records = []
    for i, (ax, task) in enumerate(zip(fig.axes, ['verbalizer', 'tg'])):
        ax.clear()
        sub = paired[paired.readout == task]
        primary = sub.drop_duplicates(['run_id', 'advisor_id', 'alpha'])
        groups = [primary.groupby('run_id').oriented_main.mean()]
        groups += [sub[sub.random_index == j].groupby('run_id').oriented_random.mean() for j in sorted(sub.random_index.unique())]
        for j, values in enumerate(groups):
            violin(ax, values, j, b.BLUE if j == 0 else b.GRAY)
            records.extend(({'readout': task, 'direction': 'Reliability' if j == 0 else f'R{j}', 'run_id': int(run), 'oriented_change': float(value), 'magnitude': 4} for run, value in values.items()))
        ax.axhline(0, color=b.INK, lw=0.75, ls=(0, (3, 3)))
        ax.set_xticks(range(6), ['Reliability', 'R1', 'R2', 'R3', 'R4', 'R5'], rotation=45, ha='right')
        ax.set_ylabel(['Oriented $\\Delta S$', 'Oriented transfer change (points)'][i], fontsize=9)
        ax.set_xlim(-0.55, 5.5)
        b.axis(ax)
    pd.DataFrame(records).to_csv(OUT / 'random_direction_distributions.csv', index=False)
    finish(fig, name)


def words():
    fig, name = capture(b.words)
    top = 4.018
    arrange(fig, name, 4.45, [top - 0.72 * 4.65, top - 0.72 * 11 / 14 * 4.65, 0.38], {0.975: 4.395, 0.014: 0.045})
    data = replay.fresh('word_level_summary')
    lex = replay.lexicon()
    limit = np.ceil(data.delta_logp.abs().max() * 10) / 10
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    cmap = LinearSegmentedColormap.from_list('probability_change', [b.RUST, '#FAFAFA', b.BLUE])
    numeric = []
    for ax, group, xpos in zip(fig.axes[:2], ['H', 'L'], [0.202, 0.701]):
        sub = data[data.word.isin(lex[f'V_{group}'])].pivot(index='word', columns='alpha', values='delta_logp').sort_index()[replay.ALPHAS]
        bounds = ax.get_position().bounds
        ax.set_position([xpos, bounds[1], 0.269, bounds[3]])
        for text in list(ax.texts):
            text.remove()
        img = ax.images[0]
        img.set_data(sub.to_numpy())
        img.set_norm(norm)
        img.set_cmap(cmap)
        img.set_extent((-0.5, 6.5, len(sub) - 0.5, -0.5))
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(len(sub) - 0.5, -0.5)
        ax.set_xticks(range(7), ['−4', '−2', '−1', '0', '+1', '+2', '+4'], fontsize=8.1)
        ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
        for word in sub.index:
            for alpha in replay.ALPHAS:
                numeric.append({'group': group, 'word': word, 'alpha': alpha, 'delta_logp': sub.loc[word, alpha]})
    colorax = fig.axes[2]
    bounds = colorax.get_position().bounds
    colorax.set_position([0.202, bounds[1], 0.768, bounds[3]])
    cb = colorax._colorbar
    cb.update_normal(plt.cm.ScalarMappable(norm=norm, cmap=cmap))
    cb.set_ticks([-limit, -limit / 2, 0, limit / 2, limit])
    fig.texts[-1].set_x(0.585)
    pd.DataFrame(numeric).to_csv(OUT / 'word_heatmap_values.csv', index=False)
    finish(fig, name)


def behavior_distribution():
    fig, name = capture(b.behavior_distribution)
    arrange(fig, name, 4.7, [2.82, 0.56], {0.97: 4.61, 0.46: 2.3})
    finish(fig, name)
