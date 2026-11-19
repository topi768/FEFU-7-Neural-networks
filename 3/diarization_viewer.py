"""Диаризация и просмотр разметки со звуком в Jupyter или Colab."""

import os
from base64 import b64encode
from functools import lru_cache
from html import escape
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Audio, HTML, display

__all__ = ["show_diarization"]


@lru_cache(maxsize=1)
def _load_pipeline():
    from pyannote.audio import Pipeline

    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=os.environ["HF_TOKEN"],
    )


def show_diarization(samples, sample_rate, diarization=None):
    """Показывает готовую разметку или запускает модель для монофонического аудио.

    При готовой разметке возвращает её, иначе возвращает результат пайплайна.
    """
    if diarization is None:
        waveform = torch.from_numpy(samples).unsqueeze(0)
        pipeline = _load_pipeline()
        result = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        diarization = result.speaker_diarization
    else:
        result = diarization

    turns = list(diarization)
    speakers = sorted({speaker for _, speaker in turns})
    speaker_rows = {speaker: row for row, speaker in enumerate(speakers)}
    duration = len(samples) / sample_rate
    time = np.arange(len(samples)) / sample_rate

    fig, (ax_audio, ax_speakers) = plt.subplots(
        2, 1, figsize=(14, 5), sharex=True, layout="constrained"
    )
    ax_audio.plot(time, samples, color="steelblue", linewidth=0.6)
    ax_audio.set_title("Звуковая волна")
    ax_audio.set_ylabel("Амплитуда")

    for turn, speaker in turns:
        row = speaker_rows[speaker]
        ax_speakers.barh(
            row,
            width=turn.end - turn.start,
            left=turn.start,
            height=0.55,
            color=f"C{row % 10}",
        )

    ax_speakers.set_yticks(range(len(speakers)), labels=speakers)
    ax_speakers.invert_yaxis()
    ax_speakers.set_title("Кто и когда говорил")
    ax_speakers.set_xlabel("Время, с")
    ax_speakers.set_xlim(0, duration)
    ax_speakers.set_axisbelow(True)
    ax_speakers.grid(axis="x", alpha=0.3)

    # После расчёта расположения осей их координаты задают область курсора.
    fig.canvas.draw()
    top_axis = ax_audio.get_position()
    bottom_axis = ax_speakers.get_position()

    image = BytesIO()
    fig.savefig(image, format="png", dpi=fig.dpi)
    plt.close(fig)

    image_base64 = b64encode(image.getvalue()).decode("ascii")
    audio = Audio(samples, rate=sample_rate)
    audio_base64 = b64encode(audio.data).decode("ascii")

    page = f"""
<audio controls preload="metadata" id="player" style="width:100%"
       src="data:audio/wav;base64,{audio_base64}"></audio>
<div style="position:relative">
  <img src="data:image/png;base64,{image_base64}"
       alt="Звуковая волна и интервалы речи"
       style="display:block; width:100%">
  <div id="timeline" style="position:absolute; cursor:ew-resize; touch-action:none;
       left:{top_axis.x0:.3%}; right:{1 - top_axis.x1:.3%};
       top:{1 - top_axis.y1:.3%}; bottom:{bottom_axis.y0:.3%}">
    <div id="cursor" style="position:absolute; inset:0 auto 0 0;
         width:3px; background:#e11d48; pointer-events:none"></div>
  </div>
</div>
<script>
const player = document.getElementById("player");
const timeline = document.getElementById("timeline");
const cursor = document.getElementById("cursor");
const duration = {duration};

function updateCursor() {{
    cursor.style.left = (100 * player.currentTime / duration) + "%";
}}

function followAudio() {{
    updateCursor();
    if (!player.paused && !player.ended) {{
        requestAnimationFrame(followAudio);
    }}
}}

function seek(event) {{
    const box = timeline.getBoundingClientRect();
    const fraction = (event.clientX - box.left) / box.width;
    player.currentTime = Math.max(0, Math.min(1, fraction)) * duration;
    updateCursor();
}}

player.addEventListener("play", followAudio);
player.addEventListener("timeupdate", updateCursor);

timeline.addEventListener("pointerdown", event => {{
    if (!Number.isFinite(player.duration)) return;
    timeline.setPointerCapture(event.pointerId);
    seek(event);
}});

timeline.addEventListener("pointermove", event => {{
    if (timeline.hasPointerCapture(event.pointerId)) seek(event);
}});
</script>
"""

    display(HTML(
        f'<iframe title="Диаризация с аудио" sandbox="allow-scripts" '
        f'style="width:100%; height:560px; border:0" '
        f'srcdoc="{escape(page, quote=True)}"></iframe>'
    ))
    return result
