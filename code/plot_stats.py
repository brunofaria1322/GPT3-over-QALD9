import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


def plot_grouped_stacked(
    dfall, path_figures, labels=None, cmap=None, filtered=False, has_legend=True
):
    """
    Plot grouped stacked bar chart of the counts of the different answers for each question.

    Parameters
    ----------
    dfall : list of pandas.DataFrame
        List of DataFrames with the counts of the different answers for each question.
    path_figures : str
        Path to the folder where the figure should be saved.
    labels : list of str, optional
        List of labels for the legend. The default is None.
    filtered : bool, optional
        If True, the error and empty answers are not included in the plot. The default is False.
    **kwargs : dict
        Additional keyword arguments for the plot.

    Returns
    -------
    None.
    """
    if filtered:
        # dfall without error and empty
        dfall = [df.drop(["error", "empty"], axis=0) for df in dfall]
        # remove last two colors from cmap
        cmap = ListedColormap(cmap.colors[:-2])

    n_df = len(dfall)
    n_types = len(dfall[0].T.columns)
    n_questions = len(dfall[0].T.index)
    H = "//"

    fig, axe = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    # print("----------------------dfall")
    for df in dfall:  # for each data frame
        # print(df.head())
        axe = df.T.plot(
            kind="bar",
            linewidth=0,
            stacked=True,
            ax=axe,
            cmap=cmap,
            legend=False,
            grid=False,
        )
    # print("----------------------out dfall")

    h, l = axe.get_legend_handles_labels()  # get the handles we want to modify

    for i in range(0, n_df * n_types, n_types):
        for _, pa in enumerate(h[i : i + n_types]):
            for rect in pa.patches:  # for each index
                """
                if path_count % 2 == 1:
                    rect.set_hatch('///')
                path_count +=1

                """
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_types))
                # rect.set_hatch(H * int((i+(n_df*n_col))**2 / n_col**3))
                rect.set_hatch(H * int(i / n_types))
                rect.set_width(1 / float(n_df + 1))

    # get max summed value in all the df in df_all and round it up to the next 10
    max_value = roundup(max([df.sum().max() for df in dfall]))
    # axe.set_xticks((np.arange(0, 2 * n_questions, 2) + 1 / float(n_df + 1)) / 2.)
    # print(np.arange(n_questions)-0.25+1 / float(n_df + 1))
    axe.set_xticks(np.arange(n_questions) - 0.25 + 1 / float(n_df + 1))
    axe.set_xticklabels(df.columns, rotation=0)
    axe.set_ylim(0, max_value)
    axe.set_ylabel("Count")
    axe.set_xlabel("Question ID")

    # Add invisible data to add another legend
    n = []
    for i in range(n_df):
        n.append(axe.bar(0, 0, color="gray", hatch=H * i))

    if labels is not None and has_legend:
        l1 = axe.legend(
            h[:n_types], l[:n_types], loc="upper left", prop={"size": 15}
        )  # , loc=[1.01, 0.5])
        axe.add_artist(l1)

        # replace "_" in labels with a "-"
        labels = [label.replace("_", "-") for label in labels]
        l2 = plt.legend(
            n, labels, loc="upper right", prop={"size": 15}
        )  # , loc=[1.01, 0.1])
        axe.add_artist(l2)
    if filtered:
        fig.savefig(
            f"{path_figures}-grouped_stacked_bar_filtered.png",
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.05,
        )
    else:
        fig.savefig(
            f"{path_figures}-grouped_stacked_bar_all.png",
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.05,
        )

    plt.close(fig)


def plot_stacked_bar(
    count_df, path_figures, cmap=None, filtered=False, has_legend=True
):
    """
    Plot stacked bar chart of the counts of the different answers for each question.

    Parameters
    ----------
    count_df : pandas.DataFrame
        DataFrame with the counts of the different answers for each question.
    path_figures : str
        Path to the folder where the figure should be saved.
    cmap : matplotlib.colors.ListedColormap
        Colormap for the plot.
    filtered : bool, optional
        If True, the error and empty answers are not included in the plot. The default is False.
    has_legend : bool, optional
        If True, the legend is included in the plot. The default is True.

    Returns
    -------
    None.
    """
    if filtered:
        # count_df without error and empty
        count_df = count_df.drop(["error", "empty"], axis=0)
        # remove last two colors from cmap
        cmap = ListedColormap(cmap.colors[:-2])

    # get max summed value in count_df and round it up to the next 10
    max_value = roundup(count_df.sum().max())

    fig, ax = plt.subplots(nrows=1, ncols=1)
    count_df.T.plot.bar(stacked=True, figsize=(10, 10), ax=ax, cmap=cmap)
    ax.set_ylim(0, max_value)
    ax.set_ylabel("Count")
    ax.set_xlabel("Question ID")
    ax.tick_params(labelrotation=0)
    if has_legend:
        plt.legend(loc="upper left", prop={"size": 15})
    else:
        plt.legend().remove()
    if filtered:
        fig.savefig(
            f"{path_figures}-stacked_bar_filtered.png",
            bbox_inches="tight",
            dpi=300,
            pad_inches=0,
        )
    else:
        fig.savefig(
            f"{path_figures}-stacked_bar_all.png",
            bbox_inches="tight",
            dpi=300,
            pad_inches=0,
        )
    plt.close(fig)


def plot_grouped_bar(
    count_df, path_figures, cmap=None, filtered=False, has_legend=True
):
    """
    Plot grouped bar chart of the counts of the different answers for each question.

    Parameters
    ----------
    count_df : pandas.DataFrame
        DataFrame with the counts of the different answers for each question.
    path_figures : str
        Path to the folder where the figure should be saved.
    cmap : matplotlib.colors.ListedColormap
        Colormap for the plot.
    filtered : bool, optional
        If True, the error and empty answers are not included in the plot. The default is False.
    has_legend : bool, optional
        If True, the legend is included in the plot. The default is True.

    Returns
    -------
    None.
    """
    if filtered:
        # count_df without error and empty
        count_df = count_df.drop(["error", "empty"], axis=0)
        # remove last two colors from cmap
        cmap = ListedColormap(cmap.colors[:-2])

    # get max value in count_df and round it up to the next 10
    max_value = roundup(count_df.max().max())

    fig, ax = plt.subplots(nrows=1, ncols=1)
    count_df.T.plot.bar(figsize=(10, 10), ax=ax, cmap=cmap)
    ax.set_ylim(0, max_value)
    ax.set_ylabel("Count")
    ax.set_xlabel("Question ID")
    ax.tick_params(labelrotation=0)
    if has_legend:
        plt.legend(loc="upper left", prop={"size": 15})
    else:
        plt.legend().remove()
    if filtered:
        fig.savefig(
            f"{path_figures}-grouped_bar_filtered.png",
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.05,
        )
    else:
        fig.savefig(
            f"{path_figures}-grouped_bar_all.png",
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.05,
        )
    plt.close(fig)


def plot_qald(count_dict, path_figures, cmap=None):
    """
    Plot pie chart of the counts of the different answers for each question.

    Parameters
    ----------
    count_dict : dict
        Dictionary with the counts of the different answers.
    path_figures : str
        Path to the folder where the figure should be saved.
    cmap : matplotlib.colors.ListedColormap
        Colormap for the plot.

    Returns
    -------
    None.
    """

    # remove error color from cmap because there is no error on QALD
    cmap = ListedColormap(cmap.colors[:-1])
    count_df = pd.DataFrame.from_dict(
        count_dict, orient="index", columns=["count"]
    ).sort_values(by="count", ascending=False)
    # sort by index
    count_df = count_df.sort_index()
    # put emptry and error in last rows
    empty_error = count_df.loc[["empty"]]
    count_df_aux = count_df.drop(["empty"])
    count_df = pd.concat([count_df_aux, empty_error])

    #########################################
    ### pie chart
    #########################################
    # custom autopct function
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            if pct > 5:
                string = f"{pct:.2f}%\n({val:d})"
            else:
                string = f"{pct:.2f}% ({val:d})"
                print(string, val)
            return string

        return my_autopct

    # plot
    plt.figure(figsize=(10, 10))
    # donut chart
    patches, texts, autotexts = plt.pie(
        count_df["count"],
        labels=count_df.index,
        autopct=make_autopct(count_df["count"]),
        colors=cmap.colors,
        wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
    )
    # autopct text position more to the edge
    for patch, txt in zip(patches, autotexts):
        # the angle at which the text is located
        ang = (patch.theta2 + patch.theta1) / 2.0
        # new coordinates of the text, 0.7 is the distance from the center
        x = patch.r * 0.8 * np.cos(ang * np.pi / 180)
        y = patch.r * 0.8 * np.sin(ang * np.pi / 180)
        txt.set_position((x, y))
    # autopct white
    for autotext in autotexts:
        autotext.set_color("white")
    # draw circle in the middle
    circle = plt.Circle((0, 0), 0.6, color="white")
    p = plt.gcf()
    p.gca().add_artist(circle)
    plt.savefig(
        f"{path_figures}_pie.png", bbox_inches="tight", dpi=300, pad_inches=0.05
    )
    plt.close()

    #########################################
    ### bar chart
    #########################################
    plt.figure(figsize=(10, 10))
    plt.bar(count_df.index, count_df["count"], color=cmap.colors)
    plt.savefig(
        f"{path_figures}_bar.png", bbox_inches="tight", dpi=300, pad_inches=0.05
    )
    plt.close()


def get_data(path, path_stats):
    """
    Get the data for the plots.

    Parameters
    ----------
    path : str
        Path to the folder with the query_dict.pkl.
    path_stats : str
        Path to the folder with the count_dict.pkl.

    Returns
    -------
    query_dict_test : dict
        Dictionary with the queries.
    count_dict : list
        List with the dictionaries with the counts of the different answers for each question.
    """
    count_dict = []

    with open(f"{path}-query_dict.pkl", "rb") as f:
        query_dict_test = pickle.load(f)

    for query_id in query_dict_test.keys():
        with open(f"{path_stats}-{query_id}-count_dict.pkl", "rb") as f:
            count_dict.append(pickle.load(f))

    # add keys to count_dict
    keys = ["boolean", "date", "number", "literal", "empty", "error", "uri"]
    for i in range(len(count_dict)):
        for key in keys:
            if key not in count_dict[i].keys():
                count_dict[i][key] = 0
    # sort count_dict by keys
    for i in range(len(count_dict)):
        count_dict[i] = {k: count_dict[i][k] for k in keys}

    count_df = pd.DataFrame([], columns=query_dict_test.keys())
    for i, query_id in enumerate(query_dict_test.keys()):
        aux_df = pd.DataFrame.from_dict(
            count_dict[i], orient="index", columns=["count"]
        )
        aux_df = aux_df.sort_values(by=["count"], ascending=False)
        count_df[query_id] = aux_df["count"]
    # sort by index
    count_df = count_df.sort_index()
    # put emptry and error in last rows
    empty_error = count_df.loc[["empty", "error"]]
    count_df_aux = count_df.drop(["empty", "error"])
    count_df_aux = pd.concat([count_df_aux, empty_error])

    return count_df_aux, count_dict


def main():
    font = {
        "family": "serif",
        #'weight' : 'bold',
        "size": 15,
    }
    matplotlib.rc("font", **font)

    engines = ["text-davinci-002", "text-davinci-003"]
    GPT3_FT = "gpt3_davinci_ft"
    fewshots = ["", "-fs5"]

    viridis = cm.get_cmap("viridis", 8)
    cmap = ListedColormap(viridis.colors)

    """
    # qald dataset
    kinds = ["train", "test"]
    for kind in kinds:
        with open(f"output/stats/qald9/qald9-{kind}-count_dict.pkl", "rb") as f:
            count_dict = pickle.load(f)
        #plot_qald(count_dict, f"figures/stats/qald9-{kind}", cmap=cmap)
    """

    # working queries
    for fs in fewshots:
        df_list = []
        dict_list = []
        for engine in engines:
            if engine == "davinci_ft":
                engine_folder_name = f"gpt3_davinci_ft"
            else:
                engine_folder_name = (
                    f"gpt3_{engine.split('-')[1]}{engine.split('-')[2]}"
                )
            path = f"output/{engine_folder_name}{fs}/{engine_folder_name}{fs}-test"
            path_stats = (
                f"output/stats/{engine_folder_name}{fs}/{engine_folder_name}{fs}-test"
            )
            path_figures = f"figures/{engine_folder_name}{fs}/{engine_folder_name}{fs}"

            count_df, count_dict = get_data(path, path_stats)
            print(count_df)
            df_list.append(count_df)
            dict_list.append(count_dict)
            plot_grouped_bar(count_df, path_figures, cmap=cmap, has_legend=True)
            plot_stacked_bar(count_df, path_figures, cmap=cmap, has_legend=True)
            plot_grouped_bar(
                count_df, path_figures, cmap=cmap, has_legend=True, filtered=True
            )
            plot_stacked_bar(
                count_df, path_figures, cmap=cmap, has_legend=True, filtered=True
            )
        plot_grouped_stacked(
            df_list,
            f"figures/stats/general{fs}",
            labels=engines,
            cmap=cmap,
            has_legend=True,
            filtered=True,
        )
        # has_legend = False

    # fine-tuned
    if engine in GPT3_FT:
        engine_folder_name = f"gpt3_davinci_ft"

        path = f"output/{engine_folder_name}{fs}/{engine_folder_name}{fs}-test"
        path_stats = (
            f"output/stats/{engine_folder_name}{fs}/{engine_folder_name}{fs}-test"
        )
        path_figures = f"figures/{engine_folder_name}{fs}/{engine_folder_name}{fs}"

        count_df, count_dict = get_data(path, path_stats)
        print(count_df)
        df_list.append(count_df)
        dict_list.append(count_dict)
        plot_grouped_bar(
            count_df,
            count_dict,
            path_figures,
            cmap=cmap,
            percent_ylim=0.65,
            has_legend=False,
        )
        plot_stacked_bar(count_df, count_dict, path_figures, cmap=cmap)
        plot_grouped_bar(
            count_df,
            count_dict,
            path_figures,
            cmap=cmap,
            filtered=True,
            percent_ylim=0.45,
        )
        plot_stacked_bar(
            count_df,
            count_dict,
            path_figures,
            cmap=cmap,
            filtered=True,
            percent_ylim=0.45,
        )


if __name__ == "__main__":
    main()
