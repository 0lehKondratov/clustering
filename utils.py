from typing import Tuple

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.metrics import (
    calinski_harabasz_score,
    silhouette_samples,
    silhouette_score,
)

from tqdm import tqdm_notebook


def calculate_optimal_distance(
    list_clusters: list, list_optimal: list
) -> Tuple[float, int]:
    """
    Поиск наибольшего расстояния для метода локтя
    :param list_clusters: список с уникальными кластерами
    :param list_optimal: список со значенями для поиска оптимальной точки на графике
    :return: макимальная дистанция, оптимальное кол-во кластеров
    """

    x1, y1 = list_clusters[0], list_optimal[0]
    x2, y2 = list_clusters[-1], list_optimal[-1]
    A, B, C = y1 - y2, x2 - x1, x1 * y2 - x2 * y1

    max_dist = -np.inf
    optimal_cluster = -np.inf
    for num, _ in enumerate(list_optimal[1:-1]):
        x_0, y_0 = list_clusters[1:-1][num], list_optimal[1:-1][num]
        dist = abs(A * x_0 + B * y_0 + C) / np.sqrt(A ** 2 + B ** 2)

        if dist > max_dist:
            max_dist = dist
            optimal_cluster = x_0
        else:
            continue
    return max_dist, optimal_cluster


def elbow_picture(
    labels_std: list,
    labels_min: list,
    labels_max: list,
    type_optimal: list,
    min_size: int,
    max_size: int,
) -> None:
    """
    Метод локтя

    Функция для вывода графика зависимостей стандартной ошибки,
    минимального и максимального числа объектов от кол-ва кластеров

    :param labels_std: список с std кол-ва объектов для разбиения на кластеры
    :param labels_min: список с min кол-вом объектов для разбиения на кластеры
    :param labels_max: список с max кол-вом объектов для разбиения на кластеры
    :param type_optimal: список для поиска оптимального значения
    :param min_size: min кол-во кластеров
    :param max_size: max кол-во кластеров
    :return: None
    """

    _, opt_cluster = calculate_optimal_distance(
        list(range(min_size, max_size + 1)), type_optimal
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        range(min_size, max_size + 1),
        labels_std,
        marker="s",
        color="green",
        label="std",
    )
    plt.plot(
        range(min_size, max_size + 1),
        labels_min,
        marker="s",
        color="grey",
        linestyle="dashed",
        label="min",
    )
    plt.plot(
        range(min_size, max_size + 1),
        labels_max,
        marker="o",
        color="grey",
        linestyle="dashed",
        label="max",
    )
    plt.xlabel("Кластер")
    plt.ylabel("Станд.ошибка / Мин.кластер / Median / Макс.кластер")
    plt.axvline(
        x=opt_cluster,
        color="black",
        label=f"optimal clust= {opt_cluster}",
        linestyle="dashed",
    )
    plt.legend()
    plt.show()


def silhouette_plot(
    data: pd.DataFrame, labels: np.array, metric="euclidean", ax: plt.Axes = None
) -> None:
    """
    Функция вывода графика силуэтного скора
    :param data: данные
    :param labels: метки кластеров
    :param metric: метрика
    :return: None
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(7, 5)

    silhouette_values = silhouette_samples(data, labels, metric=metric)
    y_lower, y_upper = 0, 0

    for i, cluster in enumerate(np.unique(labels)):
        cluster_silhouette_vals = silhouette_values[labels == cluster]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax.barh(
            range(y_lower, y_upper), cluster_silhouette_vals, edgecolor="none", height=1
        )
        ax.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
        y_lower += len(cluster_silhouette_vals)

    # Получение средней оценки силуэтного скора и построение графика
    avg_score = float(np.mean(silhouette_values))
    ax.axvline(avg_score, linestyle="--", linewidth=1, color="red")
    ax.set_xlabel(f"Silhouette  = {round(avg_score, 1)}")
    ax.set_ylabel("Метки кластеров")
    ax.set_title("График силуэта для различных кластеров", y=1.02)


def metric_picture(
    score_list: list,
    min_size: int,
    max_size: int,
    name_metric: str,
    optimal: bool = True,
):
    """
    Функция для вывода графика зависимости метрики от кол-ва кластеров
    :param score_list: список со значениями метрики
    :param min_size: min кол-во кластеров
    :param max_size: max кол-во кластеров
    :param name_metric: название метрики
    :param optimal: Нужно ли делать поиск кол-ва кластеров по методу локтя
    :return:
    """
    plt.figure(figsize=(8, 6))

    if optimal:
        _, opt_cluster = calculate_optimal_distance(
            list(range(min_size, max_size + 1)), list_optimal=score_list
        )
        plt.plot(range(min_size, max_size + 1), score_list, marker="s")
        plt.axvline(
            x=opt_cluster,
            color="black",
            label=f"optimal clust= {opt_cluster}",
            linestyle="dashed",
        )
    else:
        plt.plot(range(min_size, max_size + 1), score_list, marker="s")

    plt.xlabel("$Clusters$")
    plt.ylabel(f"${name_metric}$")
    plt.show()


def plot_size(data: pd.DataFrame, labels: np.array, ax: plt.Axes) -> None:
    """
    Фунция для вывода графика размера кластеров
    :param data: данные
    :param labels: метки кластеров
    :return: None
    """
    data = data.assign(cluster=labels)
    data = pd.DataFrame(data.groupby("cluster").count().iloc[:, 0])
    data.columns = ["value"]
    data = data.reset_index()
    sns.barplot(data=data, y="cluster", x="value", orient="h", ax=ax)
    ax.set_xlabel("Кол-во объектов")
    ax.set_title("Размер кластеров", y=1.02)


def plotting_object(data: pd.DataFrame, labels: np.array, object_cols: list) -> None:
    """
    График heatmap для сравнения признаков типа object между кластерами
    :param data: датасет
    :param labels: метки кластеров
    :param object_cols: колонки типа object
    :return: None
    """

    n_cols = len(object_cols)
    rows = n_cols // 3 + n_cols % 3

    data_label = data.assign(cluster=labels)
    size_cluster = data_label.groupby("cluster").count().iloc[:, 0]

    _, axes = plt.subplots(ncols=3, nrows=rows, figsize=(20, 4 * rows))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    for num, col in enumerate(object_cols):
        data_norm = (
            data_label.groupby(["cluster"])[col].value_counts().unstack(fill_value=0).T
            / size_cluster
        )

        data_norm = pd.DataFrame(data_norm.unstack())
        data_norm.columns = ["norm"]
        data_norm = data_norm.reset_index()

        sns.barplot(
            data=data_norm,
            x=data_norm[col],
            hue=data_norm.cluster,
            y=data_norm.norm,
            ax=axes.reshape(-1)[num],
        )


def plotting_kde_num(data: pd.DataFrame, labels: np.array, num_cols: list) -> None:
    """
    График kdeplot для сравнения признаков типа int/float между кластерами
    :param data: датасет
    :param labels: метки кластеров
    :param num_cols: колонки типа int/float
    :return: None
    """

    n_cols = len(num_cols)
    rows = n_cols // 3 + 1
    data_label = data.assign(cluster=labels)

    _, axes = plt.subplots(ncols=3, nrows=rows, figsize=(20, 4 * rows))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    for num, col in enumerate(num_cols):
        sns.kdeplot(
            data=data_label,
            x=col,
            hue="cluster",
            fill=True,
            common_norm=False,
            palette="crest",
            alpha=0.4,
            linewidth=2,
            ax=axes.reshape(-1)[num],
        )


def plotting_num(data: pd.DataFrame, labels: np.array, num_cols: list) -> None:
    """
    График kdeplot для сравнения признаков типа int/float между кластерами
    :param data: датасет
    :param labels: метки кластеров
    :param num_cols: колонки типа int/float
    :return: None
    """

    n_cols = len(num_cols)
    rows = n_cols // 3 + 1
    data_label = data.assign(cluster=labels)

    _, axes = plt.subplots(ncols=3, nrows=rows, figsize=(20, 5 * rows))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    for num, col in enumerate(num_cols):
        sns.boxplot(
            data=data_label,
            y=col,
            x="cluster",
            palette="crest",
            ax=axes.reshape(-1)[num],
        )


def plot_clustering(
    data: pd.DataFrame,
    data_scale: np.array,
    embedding: np.array,
    model: sklearn.base.ClusterMixin,
    kwargs: dict,
    min_size: int = 2,
    max_size: int = 11,
    type_train: str = None,
) -> dict:
    """
    Общий метод подбора кол-ва кластеров с выодом графиков
    :param data: данные
    :param data_scale: скалированные данные
    :param embedding: эмбединги
    :param model: модель кластеризации
    :param kwargs: параметры алгоритма
    :param min_size: min кол-во кластеров
    :param max_size: max кол-во кластеров
    :param type_train: по каким данным кластеризуем
    :return: словарь с метками кластеров
    """

    calinski_harabasz = []
    silhouette_list = []
    dict_clusters = {}

    if model().__class__.__name__ == "KMeans":
        sse = []
    else:
        labels_std = []
        labels_min = []
        labels_max = []

    for clust in tqdm_notebook(range(min_size, max_size + 1)):
        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(18, 5)


        clf = model(n_clusters=clust, **kwargs)

        if type_train == "embedding":
            clf.fit(embedding)
            
        else:
            if model().__class__.__name__ == "KMeans":
                clf.fit(data_scale)
            else:
                clf.fit(data)
        
        dict_clusters[clust] = clf.labels_

        print(clust, "clusters")
        print("-" * 100)

        if clf.__class__.__name__ == "KMeans":
            sse.append(clf.inertia_)
        else:
            # добавление статистики по размерам кластеров
            _, counts = np.unique(dict_clusters[clust], return_counts=True)
            labels_std.append(np.std(counts))
            labels_min.append(np.min(counts))
            labels_max.append(np.max(counts))

        # размер кластеров
        plot_size(data, dict_clusters[clust], ax=axes[0])
        # график silhouette
        silhouette_plot(data, dict_clusters[clust], ax=axes[1])

        # calinski_harabasz
        calinski_harabasz.append(calinski_harabasz_score(data, dict_clusters[clust]))
        # silhouette
        silhouette_list.append(silhouette_score(data, dict_clusters[clust]))
        plt.show()

    if clf.__class__.__name__ == "KMeans":
        metric_picture(sse, min_size, max_size, name_metric="SSE")
    else:
        elbow_picture(
            labels_std, labels_min, labels_max, labels_max, min_size, max_size
        )

    metric_picture(
        calinski_harabasz,
        min_size,
        max_size,
        name_metric="Calinski harabasz",
        optimal=False,
    )
    metric_picture(
        silhouette_list, min_size, max_size, name_metric="Silhouette", optimal=False
    )

    return dict_clusters