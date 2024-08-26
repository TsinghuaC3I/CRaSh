#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import srsly
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, inset_axes, mark_inset

def visualize_heatmap(data,
                      fname,
                      xlabel="layer",
                      ylabel="layer",
                      title=None,
                      fig_dir="figs5",
                      fig_size=None,
                      adjust=None,
                      sort_index=True):
    sns.set_theme()

    if sort_index:
        data.sort_index(ascending=False, inplace=True)

    if not os.path.exists(f"{fig_dir}/metadata"):
        os.makedirs(f"{fig_dir}/metadata")
    data.to_csv(f"{fig_dir}/metadata/{fname}_heatmap.csv")

    ax = None
    if fig_size is not None:
        fig, ax = plt.subplots(figsize=fig_size)

    sns.heatmap(data, ax=ax)

    if title is not None:
        plt.title(title)

    if adjust is not None:
        plt.subplots_adjust(bottom=adjust[0], left=adjust[1])

    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)

    plt.savefig(f"{fig_dir}/{fname}_heatmap.png")
    plt.savefig(f'{fig_dir}/{fname}_heatmap.pdf', format='pdf')


def visualize_plot(data,
                   fname,
                   xlabel="layer",
                   ylabel="accuracy",
                   title=None,
                   axhline=None):
    sns.set_theme()

    # data.sort_index(ascending=False, inplace=True)

    data.to_csv(f"./metadata/loss_landscape/{fname}_plot.csv")
    ax = sns.lineplot(data=data, markers=True)

    if title is not None:
        plt.title(title)

    # plt.xticks(list(data.index))
    if axhline is not None:
        ax.axhline(y=axhline["y"], xmin=data.index[0], xmax=data.index[-1])
        # axhline(y=axhline["y"],
        #             color='b',
        #             linestyle='--',
        #             label=axhline.get("label", None))
        # sns.axhline(y=axhline["y"], color="red")

    plt.xlabel(xlabel=xlabel)
    plt.ylabel(ylabel=ylabel)

    plt.savefig(f'./figures/{fname}_plot.pdf', format='pdf')


def visualize_for_interpolate():
    lines = srsly.read_jsonl(
        "../data/opt-1.3b-openbookqa-interpolate-0.02-0424.json")

    data = {"coef": [], "acc": []}

    for line in lines:
        data["coef"].append(line["coef"])
        data["acc"].append(line["results"]["openbookqa"]["acc"])
    df = pd.DataFrame(data=data, columns=["acc"], index=data["coef"])
    visualize_plot(df,
                   fname="opt-1.3b-openbookqa-interpolate-0.02",
                   xlabel="coef",
                   ylabel="acc",
                   title="OPT-1.3b OpenBookQA Interpolate")


def visualize_for_modular_crucial():
    data = {"layer": [], "acc": []}
    for layer in range(24):
        result = list(
            srsly.read_jsonl(
                "../data/opt-1.3b-openbookqa-delete-layer-%d.json" % layer))[0]
        data["layer"].append(result["delete_layer_id"])
        data["acc"].append(result["results"]["openbookqa"]["acc"])
    print(data)
    df = pd.DataFrame(data=data, columns=["acc"], index=data["layer"])
    visualize_plot(df,
                   fname="opt-1.3b-openbookqa-modular-crucial-delete",
                   xlabel="layer",
                   ylabel="acc",
                   title="OPT-1.3b OpenBookQA Modular Crucial",
                   axhline={
                       "y": 0.324,
                       "label": "fine-tuned"
                   })


def visual_for_loss_surface(df,
                            fname,
                            points=None,
                            title=None,
                            strs="start,end1,end2",
                            reverse_z=False):
    # libraries

    # Get the data (csv file is hosted on the web)
    # url = 'https://raw.githubusercontent.com/holtzy/The-Python-Graph-Gallery/master/static/data/volcano.csv'
    # data = pd.read_csv(url)

    # # Transform it to a long format
    # df = data.unstack().reset_index()
    # df.columns = ["X", "Y", "Z"]

    # # And transform the old column name in something numeric
    # df['X'] = pd.Categorical(df['X'])
    # df['X'] = df['X'].cat.codes

    # print(df)

    # Make the plot
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
    # plt.show()

    # # to Add a color bar which maps values to colors.
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf=ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.viridis, linewidth=0.2)
    # plt.show()
    if reverse_z:
        df["Z"] = -df["Z"]

    # # Rotate it
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(df['Y'],
                           df['X'],
                           df['Z'],
                           cmap=plt.cm.viridis,
                           linewidth=0.2)
    ax.view_init(45, 75)
    fig.colorbar(surf, shrink=0.5)  #, aspect=5)
    ax.set_xlabel("α")
    ax.set_ylabel("β")
    ax.set_zlabel("loss")
    plt.xticks(list(range(-4, 5)))
    plt.yticks(list(range(-4, 5)))
    ax.set_zticklabels(labels=[], rotation=30)

    if title is not None:
        ax.set_title(title)

    # plt.show()
    # # Other palette
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_trisurf(df['Y'], df['X'], df['Z'], cmap=plt.cm.jet, linewidth=0.01)
    # ax.scatter(0, 0, 5, s=10, c="r", marker="o")
    # ax.annotate('Init', xy=(0, 0), xytext=(-1.5, -1.5),
    #             arrowprops=dict(facecolor='black', shrink=0.05))

    # ax.scatter(0, 1, 0, s=10, c="r", marker="x")
    # ax.annotate('Transfer', xy=(0, 1), xytext=(1.5, 1.5),
    #             arrowprops=dict(facecolor='black', shrink=0.05))

    # ax.scatter(1, 0, 0, s=10, c="r", marker=".")
    # ax.annotate('Full FT', xy=(1, 0), xytext=(1.5, -1.5),
    #             arrowprops=dict(facecolor='black', shrink=0.05))
    if points is not None:
        # ax.plot(points["X"][0],
        #            points["Y"][0],
        #            points["Z"][0],
        #            "o",
        #            color='r',
        #            markersize=20)

        ax.text(points["X"][0],
                points["Y"][0],
                points["Z"][0],
                strs.split(",")[0],
                size=7,
                c='r')
        # ax.scatter(points["X"][0],
        #         points["Y"][0],
        #         points["Z"][0],
        #         marker="o",
        #         s=10,
        #         c='r')

        ax.text(points["X"][1],
                points["Y"][1],
                points["Z"][1],
                strs.split(",")[1],
                size=7,
                c='r')
        # ax.scatter(points["X"][1],
        #         points["Y"][1],
        #         points["Z"][1],
        #         marker="x",
        #         s=10,
        #         c='r')

        ax.text(points["X"][2],
                points["Y"][2],
                points["Z"][2],
                strs.split(",")[2],
                size=7,
                c='r')
        # ax.scatter(points["X"][2],
        #         points["Y"][2],
        #         points["Z"][2],
        #         marker="v",
        #         s=10,
        #         c='r')

        # ax.quiver(points["X"][0],
        #           points["Y"][0],
        #           points["Z"][0],
        #           points["X"][1],
        #           points["Y"][1],
        #           points["Z"][1],
        #           color=(1, 0, 0, 0.5))
        # ax.quiver(points["X"][0],
        #           points["Y"][0],
        #           points["Z"][0],
        #           points["X"][2],
        #           points["Y"][2],
        #           points["Z"][2],
        #           color=(1, 0, 0, 0.5))

    # plt.show()
    # plt.savefig(f"./figures/{fname}_loss_surface.png")
    plt.savefig(f"./figures/{fname}_loss_surface.pdf", format="pdf")


def visual_for_loss_surface_2d(data, fname, points, strs):
    fig = plt.figure()
    v = np.linspace(-4, 4, 50)
    x, y = np.meshgrid(v, v)
    z = np.array(data["Z"]).reshape(50, 50).T
    plt.contourf(x, y, z, 50)
    fig.suptitle('Input Data Contour Map')
    plt.xlabel('α')
    plt.ylabel('β')
    plt.xticks(np.linspace(-4, 4, 9))
    plt.yticks(np.linspace(-4, 4, 9))
    plt.colorbar()

    plt.scatter(0, 0, s=10, c="black", marker="o")
    plt.text(-0.6, -0.2, strs.split(",")[0], size=7, c='black')
    # plt.text(-0.6, -0.2, "start", size=7, c='black')
    # plt.annotate('Init', xy=(0, 0), xytext=(-1.5, -1.5),
    #             arrowprops=dict(facecolor='black', shrink=0.05))

    # plt.annotate('Transfer', xy=(0, 1), xytext=(1.5, 1.5),
    #             arrowprops=dict(facecolor='black', shrink=0.05))
    plt.scatter(0, 1, s=10, c="r", marker="x")
    plt.text(0.1, 1.1, strs.split(",")[1], size=7, c='r')
    # plt.text(0.1, 1.1, "end2", size=7, c='r')
    plt.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, width=0.01)

    # plt.annotate('Full FT', xy=(1, 0), xytext=(1.5, -1.5),
    #             arrowprops=dict(facecolor='black', shrink=0.05))
    plt.scatter(1, 0, s=10, c="r", marker="v")
    plt.text(1.1, 0.1, strs.split(",")[2], size=7, c='r')
    # plt.text(1.1, 0.1, "end1", size=7, c='r')
    plt.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, width=0.01)

    # plt.savefig(f"./figures/{fname}_loss_surface_2d.png")
    plt.savefig(f"./figures/{fname}_loss_surface_2d.pdf", format="pdf")
    plt.cla()


def visual_for_loss_surface_1d_2d(data,
                                  fname,
                                  points=None,
                                  title=None,
                                  strs="start,end1,end2"):
    # plt.rcParams["font.size"] = 20

    df = pd.DataFrame({k: data[k] + v for k, v in points.items()})
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_trisurf(df['Y'],
                           df['X'],
                           df['Z'],
                           cmap=plt.cm.viridis,
                           linewidth=0.2)
    ax.view_init(45, 75)
    fig.colorbar(surf, shrink=0.5)  #, aspect=5)
    ax.set_xlabel("α")
    ax.set_ylabel("β")
    ax.set_zlabel("loss")
    plt.xticks(list(range(-4, 5)))
    plt.yticks(list(range(-4, 5)))
    ax.set_zticklabels(labels=[], rotation=30)

    if title is not None:
        ax.set_title(title, fontsize=15)

    if points is not None:
        # ax.plot(points["X"][0],
        #            points["Y"][0],
        #            points["Z"][0],
        #            "o",
        #            color='r',
        #            markersize=20)

        ax.text(points["X"][0] - 0.4,
                points["Y"][0] - 0.4,
                points["Z"][0],
                strs.split(",")[0],
                size=10,
                c='r')
        ax.text(points["X"][0],
                points["Y"][0],
                points["Z"][0],
                "●",
                size=6,
                c='black')
        # ax.scatter(points["X"][0],
        #         points["Y"][0],
        #         points["Z"][0],
        #         marker="o",
        #         s=10,
        #         c='r')

        ax.text(points["X"][1] + 0.5,
                points["Y"][1] + 0.6,
                points["Z"][1],
                strs.split(",")[1],
                size=9,
                c='r')
        ax.text(
            points["X"][1],
            points["Y"][1],
            points["Z"][1],
            "✖︎",  #★
            size=6,
            c='r')
        # ax.scatter(points["X"][1],
        #         points["Y"][1],
        #         points["Z"][1],
        #         marker="x",
        #         s=10,
        #         c='r')

        ax.text(points["X"][2] + 0.5,
                points["Y"][2] - 0.8,
                points["Z"][2],
                strs.split(",")[2],
                size=10,
                c='r')
        ax.text(
            points["X"][2],
            points["Y"][2],
            points["Z"][2],
            "▼",  # 
            size=6,
            c='r')
        # ax.scatter(points["X"][2],
        #         points["Y"][2],
        #         points["Z"][2],
        #         marker="v",
        #         s=10,
        #         c='r')

        # ax.quiver(points["X"][0],
        #           points["Y"][0],
        #           points["Z"][0],
        #           points["X"][1],
        #           points["Y"][1],
        #           points["Z"][1],
        #           color=(1, 0, 0, 0.5))
        # ax.quiver(points["X"][0],
        #           points["Y"][0],
        #           points["Z"][0],
        #           points["X"][2],
        #           points["Y"][2],
        #           points["Z"][2],
        #           color=(1, 0, 0, 0.5))

    # data = {k: data[k] + v for k, v in points.items()}
    ax2d = inset_axes(ax, width="30%", height="30%", loc="upper right")

    v = np.linspace(-4, 4, 50)
    x, y = np.meshgrid(v, v)
    z = np.array(data["Z"]).reshape(50, 50).T
    ax2d.contourf(x, y, z, 50)
    # ax2d.set_xlabel('α')
    # ax2d.set_ylabel('β')
    ax2d.set_xticks(np.linspace(-4, 4, 9))
    ax2d.set_yticks(np.linspace(-4, 4, 9))
    ax2d.set_ylim([-1, 2])
    ax2d.set_xlim([-1, 2])
    # ax2d.colorbar()

    ax2d.scatter(0, 0, s=10, c="black", marker="o")
    ax2d.text(-0.1, -0.1, strs.split(",")[0], size=10, c='r')
    # ax2d.text(-0.6, -0.2, "start", size=10, c='black')
    # ax2d.annotate('Init', xy=(0, 0), xytext=(-1.5, -1.5),
    #             arrowprops=dict(facecolor='black', shrink=0.05))

    # ax2d.annotate('Transfer', xy=(0, 1), xytext=(1.5, 1.5),
    #             arrowprops=dict(facecolor='black', shrink=0.05))
    ax2d.scatter(0, 1, s=10, c="r", marker="x")
    ax2d.text(0.5, 1.5, strs.split(",")[1], size=8, c='r')
    # ax2d.text(0.1, 1.1, "end2", size=10, c='r')
    ax2d.quiver(0, 0, 0, 1, angles='xy', scale_units='xy', scale=1, width=0.01)

    # ax2d.annotate('Full FT', xy=(1, 0), xytext=(1.5, -1.5),
    #             arrowprops=dict(facecolor='black', shrink=0.05))
    ax2d.scatter(1, 0, s=10, c="r", marker="v")
    ax2d.text(1.5, -0.3, strs.split(",")[2], size=10, c='r')
    # ax2d.text(1.1, 0.1, "end1", size=10, c='r')
    ax2d.quiver(0, 0, 1, 0, angles='xy', scale_units='xy', scale=1, width=0.01)
    ax2d.tick_params(axis='both', which='both', direction='in', labelsize=8)

    # ax2d.set_xticklabels([])
    # ax2d.set_yticklabels([])
    # ax2d.set_axis_off()
    ax2d.invert_xaxis()
    ax2d.invert_yaxis()
    # mark_inset(ax, ax2d, loc1=2, loc2=4, fc="none", ec="r", ls='--')
    # plt.title(f"{task.capitalize()}", fontsize=10)
    plt.savefig(f"./figures/1d_2d_loss_surface_{fname}.pdf", format="pdf")


def test_visualize_for_loss_surface_2d():
    # visualize_for_modular_crucial()
    # task = "arc_easy"
    # task = "openbookqa"
    task = "web_questions"

    data = srsly.read_json(
        f"../figs4/metadata/opt-1.3b-{task}-distill-interp-2d.json")
    # openbookqa distill
    if task == "openbookqa":
        points = {
            'X': [0, 0, 1, 1],
            'Y': [0, 1, 0, 1],
            'Z': [6.1796875, 4.1015625, 3.71875, 4.296875]
        }
    # web_questions distill
    elif task == "web_questions":
        points = {
            'X': [0, 0, 1, 1],
            'Y': [0, 1, 0, 1],
            'Z': [3.59375, 2.63671875, 2.345703125, 2.78125]
        }
    # arc_easy
    elif task == "arc_easy":
        points = {
            'X': [0, 0, 1, 1],
            'Y': [0, 1, 0, 1],
            'Z': [4.3203125, 2.62890625, 2.171875, 2.259765625]
        }

    visual_for_loss_surface_2d(data, task, points)

    # data = {k: data[k] + v for k, v in points.items()}
    # visual_for_loss_surface(pd.DataFrame(data), "test", points=points, title="opt-1.3b, %s dataset" % task)


def test_visualize_for_loss_surface():
    for task, name in [["openbookqa", "crash-vs-full"],
                       ["openbookqa", "oft-vs-crash-vs-full"],
                       ["arc_easy", "crash-vs-full"],
                       ["arc_easy", "oft-vs-crash-vs-full"]]:
        if task == "openbookqa" and name == "crash-vs-full":
            points = {
                'X': [0, 0, 1, 1],
                'Y': [0, 1, 0, 1],
                'Z': [6.15234375, 4.27734375, 3.697265625, 4.33203125]
            }
            strs = "Init,CRaSh,Full"
            fname = f"opt-1.3b-{task}-2d-{name}"

        elif task == "openbookqa" and name == "oft-vs-crash-vs-full":
            points = {
                'X': [0, 0, 1, 1],
                'Y': [0, 1, 0, 1],
                'Z': [5.72265625, 4.27734375, 3.6953125, 4.7109375]
            }
            strs = "OFT,CRaSh,Full"
            fname = f"opt-1.3b-{task}-2d-{name}"

        elif task == "arc_easy" and name == "crash-vs-full":
            points = {
                'X': [0, 0, 1, 1],
                'Y': [0, 1, 0, 1],
                'Z': [4.3203125, 2.859375, 2.171875, 2.328125]
            }
            strs = "Init,CRaSh,Full"
            fname = f"opt-1.3b-{task}-2d-{name}"

        elif task == "arc_easy" and name == "oft-vs-crash-vs-full":
            points = {
                'X': [0, 0, 1, 1],
                'Y': [0, 1, 0, 1],
                'Z': [2.865234375, 3.390625, 2.171875, 4.65234375]
            }
            strs = "OFT,CRaSh,Full"
            fname = f"opt-1.3b-{task}-2d-{name}"

        data = srsly.read_json(
            f"/root/kyzhang/studio/transfer_llm/test/metadata/loss_landscape/opt-1.3b-{task}-{name}-interp-2d.json"
        )

        # visual_for_loss_surface_2d(data, fname, points, strs)

        data = {k: data[k] + v for k, v in points.items()}
        visual_for_loss_surface(pd.DataFrame(data),
                                fname,
                                strs=strs,
                                points=points)


if __name__ == "__main__":
    for task in [
            "arc_easy", "arc_challenge", "web_questions", "openbookqa", "piqa",
            "hellaswag", "sciq", "race"
    ]:
        for name in ["crash-vs-full", "oft-vs-crash-vs-full"]:
            try:
                points = srsly.read_json(
                    f"./metadata/loss_landscape/opt-1.3b-{task}-{name}-interp-2d_4points.json"
                )
                data = srsly.read_json(
                    f"./metadata/loss_landscape/opt-1.3b-{task}-{name}-interp-2d.json"
                )
                # data = {k: data[k] + v for k, v in points.items()}
                strs = "OFT,CRaSh,Full" if "oft" in name else "Init,CRaSh,Full"
                fname = f"{task}-{name}"
                visual_for_loss_surface_1d_2d(data,
                                              fname,
                                              strs=strs,
                                              points=points,
                                              title=task)
                print(f"Done {task}-{name}")
            except Exception as e:
                print(e)