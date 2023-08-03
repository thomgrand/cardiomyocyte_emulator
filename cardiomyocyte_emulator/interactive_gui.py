from pathlib import Path
from typing import Iterable, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Slider, Button, RadioButtons
import torch
from .emulator import load_default_emulator_model, APEmulator
import pandas as pd
from functools import partial
import re

#Original demo:
#https://matplotlib.org/2.0.2/users/screenshots.html#slider-demo

def setup_fig(t : np.ndarray, trace : np.ndarray, slider_names : Iterable[str], slider_ranges : Iterable[Tuple[int, int]], sliders_space : float = 0.4, outline_data : Iterable[Tuple[np.ndarray, np.ndarray]]=None) -> Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, sharey=True)
    #fig, ax = plt.subplots()

    if outline_data is not None:
        for data in outline_data:
            ax1.plot(data[0], data[1], linestyle="--")
            ax2.plot(data[0], data[1], linestyle="--")

    trace_left_h = ax1.plot(t, trace, color="k", linewidth=1.25)
    trace_right_h = ax2.plot(t, trace, color="k", linewidth=1.25)
    ax2.set_xlim(-0.5, 2.5)

    #slider_space = 0.5
    plt.subplots_adjust(bottom=sliders_space+0.1)
    slider_height = sliders_space / len(slider_names)
    axes = [ax1, ax2]
    sliders = []
    for i, n in enumerate(slider_names):
        r = (slider_ranges[0][i], slider_ranges[1][i])
        slider_axis = plt.axes([0.25, slider_height * i, 0.65, slider_height], facecolor="b")
        sliders.append(Slider(slider_axis, n, r[0], r[1], valinit=(r[0] + r[1])/2, valfmt="%.1e"))
        axes.append(slider_axis)

    #Prettify
    ax1.set_xlabel("Time [ms]")
    ax2.set_xlabel("Time [ms]")
    ax1.set_ylabel("$V_m$ [mV]")
    
    return fig, axes, sliders, (trace_left_h, trace_right_h)

def interactive_plot(emulator : APEmulator, param_path : Path):
    t = np.concatenate([np.linspace(-5, 5, num=100)[:-1], np.linspace(5, 500, num=400)])
    #info = np.load("trace_study/biomarker_trace_infos.npz")
    #t = info["t"]
    #baseline, drug = import_paper_drug_plot()
    #drug_df, drug_list = import_hungarian_drug_data(drug_name="Dofetilide.*")
    
    #drug_params = pd.read_csv(param_path, sep="\t")
    #drug_params = drug_params[np.any(np.array([[re.match(f"{emulator}.*", c) is not None for c in  drug_df.columns] for emulator in drug_params.emulator]), axis=-1)]
    #assert len(drug_list) == len(drug_params)
    #model_names = drug_params["emulator"]
    max_conds_names = [re.sub("_b$", "", n) for n in emulator.max_conds_names][::-1]
    fig, axes, sliders, traces_h = setup_fig(t, emulator.forward_latent_np(t, emulator.max_conds_center)[0], 
                                             max_conds_names, np.stack(emulator.max_conds_ranges)[:, ::-1], 
                    outline_data=[]) #[(drug_list[i]["t"] + drug_params.iloc[i]["t_offset"], drug_list[i]["Vm"] - drug_params.iloc[i]["Vm_offset"]) for i in range(len(drug_list))])

    def update(val):
        param_vec = np.array([s.val for s in sliders])
        trace = emulator.forward_latent_np(t, param_vec[::-1])[0]
        [t[0].set_ydata(trace) for t in traces_h]
        fig.canvas.draw_idle()

    [s.on_changed(update) for s in sliders]

    resetax = plt.axes([0.05, 0.05, 0.1, 0.04])
    button_reset = Button(resetax, 'Reset', hovercolor='0.975')
    button_reset.on_clicked(lambda event: [s.reset() for s in sliders])

    button_axes = []
    buttons = []
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #callback_general = lambda event, i: [s.set_val(drug_params.iloc[i][b_n]) for s, b_n in zip(sliders[::-1], emulator.biomarker_names)]
    """
    for drug_i in range(len(drug_list)):
        button_axes.append((button_ax := plt.axes([0.05, 0.15 + (drug_i*0.1), 0.1, 0.04])))
        buttons.append((button := Button(button_ax, drug_params.iloc[drug_i]["emulator"], color=color_cycle[drug_i], hovercolor='0.975')))  
        callback_f = partial(callback_general, i=drug_i)
        button.on_clicked(callback_f)
    """
    
    #plt.show()
    return fig

if __name__ == "__main__":
    #TODO: Move somewhere else
    emulator = load_default_emulator_model()
    interactive_plot(emulator, None)
    plt.show()
