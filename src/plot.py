"""Render all seven statistical figures from the recomputed record-level tables."""
import re
from .paths import OUTPUT
from .plots import components as b, layout, alignment
import matplotlib.pyplot as plt
from matplotlib.text import Text

def main():
    (OUTPUT/'figures').mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],
        'font.size':9,'mathtext.fontset':'cm','axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
        'text.color':b.INK,'axes.labelcolor':b.INK,'axes.edgecolor':b.INK,'axes.linewidth':.65,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
    original=b.export
    def export(fig,name):
        transform={'f2_behavior':alignment.behavior,'f4_steering':alignment.steering,'s2_word_changes':alignment.words}.get(name)
        if transform:transform(fig)
        for artist in fig.findobj(match=Text):
            s=artist.get_text().replace('Partner description','Source description').replace('queried partner ID','queried ID')
            s=s.replace("the same partner's query state","the same identity's query state").replace('partner queries','identity queries')
            artist.set_text(re.sub(r'\bpartner\b','participant',re.sub(r'\bpartners\b','participants',s)))
        original(fig,name)
    b.export=export
    for fn in [layout.behavior,layout.descriptions,layout.steering,layout.patching,layout.random_controls,layout.words,layout.behavior_distribution]:fn()
    print('Rendered 7 figures in outputs/figures (PDF, SVG, PNG).')

if __name__=='__main__':main()
