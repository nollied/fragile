import tempfile

import panel
from PIL import Image


def create_gif(data, filename=None, fps=10, optimize=False):
    duration = int((len(data) / fps) * 20)
    filename = tempfile.NamedTemporaryFile(suffix="a.gif") if filename is None else filename
    images = [Image.fromarray(v) for v in data]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        optimize=optimize,
        duration=duration,
        loop=0,
    )
    return filename


def show_root_game(swarm, fps=10, optimize=False):
    vals, *_ = zip(*list(swarm.tree.iterate_root_path(names=["rgb"])))
    f = create_gif(vals, fps=fps, optimize=optimize)
    return panel.pane.GIF(f)
