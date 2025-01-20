import psutil, datetime, os

def set_graph(ax, title, ylabel, xlabel):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid()

def ram_usage(file_path: (str | None) = None, store: bool = True) -> float:
    ram_used_GB = psutil.virtual_memory()[3]/1000000000
    if store:
        if file_path is not None:
            if os.path.exists(file_path) is False:
                with open(file_path, "w") as f:
                    f.write("RAM usage\n")

            with open(file_path, "a") as file:
                file.write(f"Time: {datetime.datetime.now()}. RAM memory % used: {psutil.virtual_memory()[2]}. RAM memory used (GB): {ram_used_GB}\n")
                
    return ram_used_GB