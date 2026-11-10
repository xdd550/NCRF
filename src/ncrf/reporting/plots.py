"""Plot saved arrays without recomputing predictions or modifying metrics."""
from pathlib import Path
import numpy as np


def plot_prediction(path, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # Explicit failure is preferable to silently substituting another font.
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family":"Times New Roman", "svg.fonttype":"none", "font.size":11})
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    with np.load(path,allow_pickle=False) as data:
        prediction,truth,control=data["prediction"],data["truth"],data["control"]
    columns=sorted(set([0,len(truth)//2,len(truth)-1]))
    fig,axes=plt.subplots(4,len(columns),figsize=(2.2*len(columns)+1.3,8),squeeze=False,layout="constrained")
    error_limit=float(np.max(np.abs(prediction-truth))) or 1e-8
    for j,k in enumerate(columns):
        axes[0,j].set_title(f"k = {k+1}")
        for i,field in enumerate((control[k],truth[k],prediction[k],np.abs(prediction[k]-truth[k]))):
            image=axes[i,j].imshow(field,origin="lower",cmap="RdBu_r" if i<3 else "magma",vmin=-1 if i<3 else 0,vmax=1 if i<3 else error_limit)
            axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
            if j==0:
                axes[i,j].set_ylabel(("Control","Reference","Prediction","Absolute error")[i])
            if j==len(columns)-1:
                fig.colorbar(image,ax=axes[i,:].tolist(),shrink=.9)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=300)
    plt.close(fig)
    return str(output)
