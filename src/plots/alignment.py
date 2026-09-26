"""Content alignment and uncertainty-band rendering."""
from . import components as b
from matplotlib.transforms import Bbox
CHANGES={}


def content_bounds(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return Bbox.union([ax.get_tightbbox(renderer) for ax in axes]).transformed(fig.transFigure.inverted())


def center_axes(fig, axes):
    bounds = content_bounds(fig, axes)
    shift = 0.5 - (bounds.x0 + bounds.x1) / 2
    for ax in axes:
        x, y, w, h = ax.get_position().bounds
        ax.set_position([x + shift, y, w, h])
    return content_bounds(fig, axes)


def center_heading(fig, index, center):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label, title = fig.texts[index:index + 2]
    widths = [text.get_window_extent(renderer).width / fig.bbox.width for text in (label, title)]
    gap = 0.018
    left = center - (sum(widths) + gap) / 2
    label.set_x(left)
    title.set_x(left + widths[0] + gap)


def behavior(fig):
    for label in fig.axes[0].get_yticklabels():
        label.set_weight('normal')
    for ax, left in zip(fig.axes, [0.25, 0.665]):
        _, bottom, width, height = ax.get_position().bounds
        ax.set_position([left, bottom, width, height])
    bounds = center_axes(fig, fig.axes)
    centers = []
    for i, ax in enumerate(fig.axes):
        panel = content_bounds(fig, [ax])
        center = (panel.x0 + panel.x1) / 2
        centers.append(center)
        center_heading(fig, 2 * i, center)
    legend = fig.legends[0]
    handles = legend.legend_handles
    labels = [text.get_text() for text in legend.get_texts()]
    legend.remove()
    fig.legend(handles=handles, labels=labels, loc='upper center', bbox_to_anchor=(centers[0], 3.57 / 3.8), ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.0, fontsize=8.4)
    fig.texts[-1].set_x(0.5)
    fig.texts[-1].set_ha('center')
    CHANGES['f2_behavior'] = {'all_model_labels_normal_weight': True, 'content_left': float(bounds.x0), 'content_right': float(bounds.x1), 'content_center': float((bounds.x0 + bounds.x1) / 2), 'title_centers': centers, 'panel_heights_inches': [ax.get_position().height * fig.get_figheight() for ax in fig.axes]}


def steering(fig):
    for ax in fig.axes:
        ax.collections[0].set_alpha(0.19)
        ax.collections[0].set_linewidth(0)
        assert len(ax.lines) == 4
        ax.lines[0].remove()
        ax.lines[0].remove()
        ax.lines[0].set_linewidth(1.7)
        assert len(ax.lines) == 2
    CHANGES['f4_steering'] = {'band_alpha': 0.19, 'boundary_lines': 0, 'mean_line_width': 1.7, 'confidence_limits_unchanged': True}


def words(fig):
    height, top, row_height, grid_width = (3.44, 2.98, 0.165, 0.3)
    fig.set_size_inches(b.WIDTH, height)
    axes = fig.axes[:2]
    for ax, left, count in zip(axes, [0.18, 0.67], [14, 11]):
        ax.set_position([left, (top - count * row_height) / height, grid_width, count * row_height / height])
    bounds = center_axes(fig, axes)
    title_centers = []
    for i, ax in enumerate(axes):
        panel = content_bounds(fig, [ax])
        center = (panel.x0 + panel.x1) / 2
        title_centers.append(center)
        center_heading(fig, 2 * i, center)
        fig.texts[2 * i].set_y(3.39 / height)
        fig.texts[2 * i + 1].set_y(3.38 / height)
    fig.axes[2].set_position([0.14, 0.38 / height, 0.72, 0.1 / height])
    fig.texts[-1].set_position((0.5, 0.045 / height))
    fig.texts[-1].set_ha('center')
    cell_width = grid_width * b.WIDTH / 7
    CHANGES['s2_word_changes'] = {'content_left': float(bounds.x0), 'content_right': float(bounds.x1), 'content_center': float((bounds.x0 + bounds.x1) / 2), 'title_centers': title_centers, 'colorbar_center': 0.5, 'cell_width_inches': cell_width, 'cell_height_inches': row_height, 'cell_width_to_height': cell_width / row_height, 'data_and_color_scale_unchanged': True}
