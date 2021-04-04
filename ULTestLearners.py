#!/usr/bin/env python3
import os
import math
import sys
import getopt
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import datasets
from sklearn.datasets import make_blobs

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split  # Import train_test_split function

from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import log_loss, classification_report
from sklearn.manifold import Isomap

# scaling the data
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA, FastICA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from scipy import stats
import scipy.cluster.hierarchy as sch

import ULClustering as ulc
import ULDimensionReduction as uld
import SUNeuralNetworks as snn

#############################################################################################################
plot_num = 0


def ret_plot_num():
    global plot_num
    plot_num = plot_num + 1
    return plot_num


#############################################################################################################
# Referred: https://wil.yegelwel.com/cluster-correlation-matrix/
def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to eachother

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage,
                                        cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]


#############################################################################################################
def plot_data(save_fold, fp, dr, x_name, x_value, score_train, score_test, cmm):
    plt.figure()
    plt.title("Wine-Q " + "-{}-".format(dr) + "NN-tuning-" + x_name)
    plt.xlabel(x_name)
    plt.ylabel("Score")
    plt.gca().invert_yaxis()
    plt.grid()
    # plot the average training and test score lines at each training set size
    plt.plot(x_value, score_train, 'o-', color="r")
    plt.plot(x_value, score_test, 'o-', color="g")
    plt.legend(["Training score", "Cross-validation score"])
    # shows scores from 0 to 1.1
    plt.ylim(-.1, 1.1)
    filename = save_fold + '{}-{}-{}-tuning.png'.format(ret_plot_num(), dr, x_name)
    print("Plot " + filename + " Created")
    fp.write("Plot " + filename + " Created" + "\n")
    plt.savefig(save_fold + '{}-{}-{}'.format(ret_plot_num(), dr, x_name) + "Wine-Q.png")
    # plt.show()
    plt.close()
    for m, cm in enumerate(cmm):
        for line in cm:
            fp.write("".join(str(line)) + "\n")
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        cax1 = ax1.matshow(cm)
        plt.title('Confusion matrix - NN - Wine-Q for ' + '{}'.format(dr))
        fig1.colorbar(cax1)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(save_fold + '{}-{}-NN-WineQ-{}'.format(ret_plot_num(), m, dr)+'confusion_matrix.png')
        # plt.show()
        plt.close()


#############################################################################################################
def plot_learning_curve(dr, fp, save_fold, plot_pfix, su_estimator, xx_train, yy_train):
    train_sizes, train_scores, test_scores = model_selection.learning_curve(estimator=su_estimator,
                                                                            X=xx_train,
                                                                            y=yy_train,
                                                                            train_sizes=np.linspace(0.1, 1.0, 5),
                                                                            cv=5)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure()
    plt.title(plot_pfix + "-" + " Neural Network {}".format(dr) + " Learning Curve")
    plt.xlabel("train set size")
    plt.ylabel("Score")
    plt.gca().invert_yaxis()
    plt.grid()
    # plot the std deviation as a transparent range at each training set size
    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1,
                     color="g")

    # plot the average training and test score lines at each training set size
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g")
    plt.legend(["Training score", "Cross-validation score"])
    # shows scores from 0 to 1.1
    plt.ylim(-.1, 1.1)
    filename = save_fold + '\\' + '{}-{}-{}-LearningCurve.png'.format(ret_plot_num(), plot_pfix, dr)
    print("Plot " + filename + " Created")
    fp.write("Plot " + filename + " Created" + "\n")
    plt.savefig(save_fold + '{}-{}-{}-LearningCurve.png'.format(ret_plot_num(), plot_pfix, dr))
    # plt.show()
    plt.close()


#############################################################################################################
def plot_help_cm(ytest, ypred, fold, pname, algo, fpt, dr):
    cm = metrics.confusion_matrix(ytest, ypred)
    # print(cm)
    for line in cm:
        fpt.write("".join(str(line)) + "\n")
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    cax1 = ax1.matshow(cm)
    plt.title('Confusion matrix - ' + '{}-{}-{}'.format(algo, pname, dr))
    fig1.colorbar(cax1)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(fold + '{}-{}-{}-{}'.format(ret_plot_num(), pname, algo, dr)+'confusion_matrix.png')
    # plt.show()
    plt.close()


#############################################################################################################
def plot_help_isomap(func, X, y, n, fn, plot_name, algo, dr):
    # Iso map
    # Create an isomap and fit the `digits` data to it
    projection = Isomap(n_neighbors=n).fit_transform(X)
    # Compute cluster centers and predict cluster index for each sample
    tclusters = func.transform_fit(X)

    # plot the results
    plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,
                c=tclusters, cmap=plt.cm.get_cmap('cubehelix', 10))
    plt.colorbar(ticks=range(n), label='{} label'.format(plot_name))
    plt.clim(-0.5, 9.5)
    plt.title("{}-{}-{}".format(plot_name, algo, dr) + "-Predicted Labels")
    plt.savefig(fn + '{}-{}-{}-{}'.format(ret_plot_num(), plot_name, algo, dr) + '-predict-heatmap.png')
    # plt.show()
    plt.close()
    plt.scatter(projection[:, 0], projection[:, 1], lw=0.1,
                c=y, cmap=plt.cm.get_cmap('cubehelix', 10))
    plt.colorbar(ticks=range(n), label='{} label'.format(plot_name))
    plt.clim(-0.5, 9.5)
    plt.title("{}-{}-{}".format(plot_name, algo, dr) + "-Training Labels")
    plt.savefig(fn + '{}-{}-{}-{}'.format(ret_plot_num(), plot_name, algo, dr) + '-train-heatmap.png')
    # plt.show()
    plt.close()


#############################################################################################################
def data_analysis_find_K(X, fol_name, plot_name):
    inertia_array = list()
    kvalues = range(1, 16)  # 1 to 15
    for k in kvalues:
        model = ulc.ULClustering(ul_algo="Kmeans", init="k-means++", n_clusters=k, initp='kmeans')
        model.add_evidence(X)
        inertia_array.append(model.ret_inertia())
    # sns.pointplot(x=list(range(2, n_clust)), y=inertia_array)
    plt.figure(figsize=(20, 6))
    plt.title('SSE on K-Means based on number of clusters')
    plt.plot(kvalues, inertia_array, marker = "o")
    plt.xticks(kvalues)
    plt.xlabel("K clusters")
    plt.ylabel("SSE")
    plt.savefig(fol_name + '{}-{}'.format(ret_plot_num(), plot_name) + '-Elbow.png')
    # plt.show()
    plt.close()


#############################################################################################################
def accuracy_test(fpt, y_train, train_pred_y, y_test, test_pred_y, dr):
    train_score = metrics.accuracy_score(y_train, train_pred_y)
    test_score = metrics.accuracy_score(y_test, test_pred_y)
    confu_matrix = metrics.confusion_matrix(y_test, test_pred_y)
    if fpt is not None:
        fpt.write("\n\n")
        fpt.write("Train Data Prediction Accuracy {}".format(dr) + " %: ")
        fpt.write("".join(str(train_score * 100)) + "\n")
        fpt.write("Test Data Validation Accuracy {}".format(dr) + " %: ")
        fpt.write("".join(str(test_score * 100)) + "\n")
        fpt.write("Test Data Confusion Matrix {}".format(dr) + " : " + "\n")
        for line in confu_matrix:
            fp.write("".join(str(line)) + "\n")
    return train_score, test_score, confu_matrix


#############################################################################################################
def metrics_log(fpt, trX, y_train, train_pred_y, valX, y_test, y_pred, algo, dr):
    homo_score = homogeneity_score(y_train, train_pred_y)
    comp_score = completeness_score(y_train, train_pred_y)
    vm_score = v_measure_score(y_train, train_pred_y)
    adj_rand_score = adjusted_rand_score(y_train, train_pred_y)
    adj_mut_score = adjusted_mutual_info_score(y_train, train_pred_y)
    sil_score = silhouette_score(trX, train_pred_y, metric='euclidean')

    if fpt is not None:
        fpt.write("Training Data Metrics -{}-{}:".format(algo, dr) + "\n")
        fpt.write('% 9s' % 'homo    compl   v-meas  ARI     AMI     silhouette' + "\n")
        fpt.write('%.5f %.5f %.5f %.5f %.5f %.5f'
              % (homo_score, comp_score, vm_score, adj_rand_score, adj_mut_score, sil_score))
        fpt.write("\n")
        # print('% 9s' % 'homo   compl  v-meas   ARI   AMI  silhouette')
        # print('%.5f   %.5f   %.5f   %.5f   %.5f    %.5f'
        #       % (homo_score, comp_score, vm_score, adj_rand_score, adj_mut_score, sil_score))

    homo_score = homogeneity_score(y_test, y_pred)
    comp_score = completeness_score(y_test, y_pred)
    vm_score = v_measure_score(y_test, y_pred)
    adj_rand_score = adjusted_rand_score(y_test, y_pred)
    adj_mut_score = adjusted_mutual_info_score(y_test, y_pred)
    sil_score = silhouette_score(valX, y_pred, metric='euclidean')

    if fpt is not None:
        fpt.write("Validation Data Metrics -{}-{}:".format(algo, dr) + "\n")
        fpt.write('% 9s' % 'homo    compl   v-meas  ARI     AMI     silhouette' + "\n")
        fpt.write('%.5f %.5f %.5f %.5f %.5f %.5f'
              % (homo_score, comp_score, vm_score, adj_rand_score, adj_mut_score, sil_score))
        fpt.write("\n")
        # print('% 9s' % 'homo   compl  v-meas   ARI   AMI  silhouette')
        # print('%.5f   %.5f   %.5f   %.5f   %.5f    %.5f'
        #       % (homo_score, comp_score, vm_score, adj_rand_score, adj_mut_score, sil_score))


#############################################################################################################
def nn_learner(data, f_name, fp, dr):
    x_train, x_test, y_train, y_test = data[0], data[1], data[2], data[3]
    # 5. Neural Networks
    print("Neural Networks")
    fp.write("\n" + "\n" + "Neural Networks" + "\n")
    # hid_layers = [(10, 10)]
    hid_layers = [(10, 10), (10, 20), (10, 10, 10)]
    alphas = 0.0005
    episodes = [1000, 1500, 2000]
    scores = {}
    scores_list = []
    x_val = []
    for element in hid_layers:
        for episode in episodes:
            val = (element, episode)
            x_val.append(val)
    # print(x_value)
    x_value = [i for i, j in enumerate(x_val)]
    x_name = "Episodes [(1000,1500,2000)]-HiddenLayers[(10,10),(10,20),(10, 10, 10)]"
    tr_scores = []
    te_scores = []
    c_matrixs = []
    saved_n = False
    print("Neural Networks - Classification Report")
    fp.write("Neural Networks - Classification Report" + "{}".format(dr) + "\n")
    for i, hl in enumerate(hid_layers):
        for j, episode in enumerate(episodes):
            val = (hl, episode)
            # create a learner and train it
            learner = snn.SUNeuralNetworks(hidden_layer=hl,
                                           episode=episode)
            learner.add_evidence(x_train, y_train)  # train it
            # Predict for test data
            pred_y = learner.model_prediction(x_test)  # get the predictions
            pred_train_y = learner.model_prediction(x_train)  # get the predictions
            tr_score, te_score, c_matrix = accuracy_test(fp, y_train, pred_train_y, y_test, pred_y, dr)
            tr_scores.append(tr_score)
            te_scores.append(te_score)
            c_matrixs.append(c_matrix)

            if not saved_n:
                mod = ("NeuralNetworkClassifier", MLPClassifier())
                plot_learning_curve(dr,
                                    fp,
                                    save_fold=f_name,
                                    plot_pfix="wine-Q",
                                    su_estimator=mod[1],
                                    xx_train=x_train,
                                    yy_train=y_train)
                saved_n = True

            scores[i, j] = te_score
            scores_list.append(te_score)
            class_report = metrics.classification_report(y_test, pred_y)
            print(class_report)

            # print("Classification report:", class_report)
            xval = "((HiddenLayer),episode) = " + str(val)
            # print(xval)
            fp.write("Classification report: ")
            fp.write("".join(xval) + "\n")
            for line in class_report:
                fp.write("".join(line))
            fp.write("\n" + "Attributes of the Learned Classifier" + "\n")
            fp.write("current loss computed with the loss function: ")
            fp.write("".join(str(learner.model.loss_)) + "\n")
            fp.write("number of iterations the solver: ")
            fp.write("".join(str(learner.model.n_iter_)) + "\n")
            fp.write("num of layers: ")
            fp.write("".join(str(learner.model.n_layers_)) + "\n")
            fp.write("Num of o/p: ")
            fp.write("".join(str(learner.model.n_outputs_)) + "\n")
    plot_data(f_name, fp, dr, x_name, x_value, tr_scores, te_scores, c_matrixs)


#############################################################################################################
# k-means clustering
def kmeans_clustering(fptr, clusters, s_data, f_name, plt_name, dr):
    x_train, x_test, y_train, y_test = s_data[0], s_data[1], s_data[2], s_data[3]
    print("K-means Clustering for " + "{}-{}".format(plt_name, dr))
    fptr.write("\n" + "\n" + "K-means Clustering for " + "{}-{}".format(plt_name, dr) + "\n")

    # 1. Init the model
    model = ulc.ULClustering(ul_algo="Kmeans",
                             init="k-means++",
                             n_clusters=clusters,
                             initp='kmeans')
    # 2. Train the model
    model.add_evidence(x_train)
    # 3. Predict from the model
    train_y_pred = model.prediction(x_train)  # Cluster labels
    # 4. Predict from the model
    val_y_pred = model.prediction(x_test)  # Cluster labels

    # Isomap
    plot_help_isomap(model,
                     x_train,
                     y_train,
                     clusters,
                     f_name,
                     plt_name,
                     "kmeans",
                     dr)
    # metrics
    metrics_log(fptr,
                x_train,
                y_train,
                train_y_pred,
                x_test,
                y_test,
                val_y_pred,
                "kmeans",
                dr)
    # confusion matrix
    plot_help_cm(y_test,
                 val_y_pred,
                 f_name,
                 plt_name,
                 "kmeans",
                 fptr,
                 dr)


#############################################################################################################
# Expectation Maximization clustering
def em_clustering(fptr, clusters, s_data, f_name, plt_name, dr):
    x_train, x_test, y_train, y_test = s_data[0], s_data[1], s_data[2], s_data[3]
    print("Expectation-Maximization Clustering for " + "{}-{}".format(plt_name, dr))
    fptr.write("\n\n" + "Expectation-Maximization Clustering for " + "{}-{}".format(plt_name, dr) + "\n")

    # 1. Init the model
    model = ulc.ULClustering(ul_algo="EM",
                             n_clusters=clusters,
                             initp='kmeans')
    # 2. Train the model
    model.add_evidence(x_train)
    # 3. Predict from the model
    train_y_pred = model.prediction(x_train)  # Cluster labels
    # 4. Predict from the model
    val_y_pred = model.prediction(x_test)  # Cluster labels

    # Isomap
    plot_help_isomap(model,
                     x_train,
                     y_train,
                     clusters,
                     f_name,
                     plt_name,
                     "EM",
                     dr)
    # # metrics
    metrics_log(fptr,
                x_train,
                y_train,
                train_y_pred,
                x_test,
                y_test,
                val_y_pred,
                "EM",
                dr)
    # confusion matrix
    plot_help_cm(y_test,
                 val_y_pred,
                 f_name,
                 plt_name,
                 "EM",
                 fptr,
                 dr)


#############################################################################################################
def usage():
    print("python SUTestLearners.py -s <os_type>")
    print("Results of the experiments are saved (text file and plots)"
          "in results/, results/digits/, results/wine-Q folders")


#############################################################################################################
if __name__ == "__main__":
    start = time.time()
    print("--- Unsupervised Learning ---" + "\n" + "\n")
    pd.plotting.register_matplotlib_converters()
    # sns.set()

    folder = "result"
    if not os.path.exists(folder):
        os.makedirs(folder)

    try:
        fname='./{}/Results.txt'.format(folder)
        os.remove(fname)
    except OSError:
        pass

    dirname = './{}/'.format(folder)
    fname = './{}/Results.txt'.format(folder)
    fp = open(fname, "a+")
    fp.write("--- Unsupervised Learning ---" + "\n" + "\n")

    # fp.write("--- Iris Dataset ---" + "\n" + "\n")
    # iris = sns.load_dataset("iris")
    # print(iris.head())
    # fp.write(iris.head())
    # sns.pairplot(iris, hue='species', height=2.5)
    # # print('iris-head=')
    # # print(iris.head())
    # # sns.FacetGrid(iris, hue="species", height=6).map(plt.scatter, 'sepal_length', 'petal_length')
    # iris_X = iris.drop('species', axis=1)
    # # # Applying Standard scaling to get optimized result
    # sc = StandardScaler()
    # irisX = sc.fit_transform(iris_X)
    # #
    # # print('iris-pop=')
    # # iris_y = iris.pop('species')
    # # print(iris.head())
    # itrain_x, itest_x, itrain_y, itest_y = train_test_split(irisX, iris['species'], test_size=0.2, random_state=7)

    # digits_df = sns.load_dataset("mnist_784")
    digits_df, labels = datasets.load_digits(return_X_y=True)
    # sns.pairplot(digits_df, hue='species', height=2.5)
    digits_X = digits_df
    digits_y = labels
    digits = digits_X

    # digits[]
    # print(digits_X)
    # print(digits_y)
    # # Applying Standard scaling to get optimized result
    sc = StandardScaler()
    digitsX = sc.fit_transform(digits_X)
    # Number of Training labels
    n_clusters2 = len(np.unique(digits_y))
    dtrain_x, dtest_x, dtrain_y, dtest_y = train_test_split(digitsX, digits_y, test_size=0.4, random_state=7)

    csv_file = './Data/winequality-white.csv'
    wine_df = pd.read_csv(csv_file, sep=';')
    # print(wine_df.head())
    # sns.pairplot(wine_df, hue='quality', height=2.5)
    wine_X = wine_df.drop('quality', axis=1)
    samples = wine_df['quality']
    # output = [1 if sample > 5 else 0 for sample in samples]
    # y = wine_df.pop('quality')
    y = wine_df['quality']
    # # Applying Standard scaling to get optimized result
    sc = StandardScaler()
    wineX = sc.fit_transform(wine_X)
    # Number of Training labels
    n_clusters1 = len(np.unique(samples))
    wtrain_x, wtest_x, wtrain_y, wtest_y = train_test_split(wineX, y, test_size=0.4, random_state=5)

    # plot_prefix = ["wine-Q", "iris"]
    # plot_prefix = ["digits", "wine-Q"]
    data_analysis_set = [wine_X, digits_X]

    # dataset = [(wtrain_x, wtest_x, wtrain_y, wtest_y), (itrain_x, itest_x, itrain_y, itest_y)]
    # dataset = [(dtrain_x, dtest_x, dtrain_y, dtest_y), (wtrain_x, wtest_x, wtrain_y, wtest_y)]
    plot_prefix = ["wine-Q", "digits"]
    dataset = [(wtrain_x, wtest_x, wtrain_y, wtest_y), (dtrain_x, dtest_x, dtrain_y, dtest_y)]
    n_clusters = [n_clusters1, n_clusters2]

    for idx, s_data in enumerate(dataset):
        fp.write("\n\n"+"Results for Data set =")
        fp.write(" ".join(plot_prefix[idx]) + "\n")
        folder_name = dirname+'{}/'.format(plot_prefix[idx])
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        clusters = n_clusters[idx]

        x_train, x_test, y_train, y_test = s_data[0], s_data[1], s_data[2], s_data[3]

        if plot_prefix[idx] == "digits":
            data_analysis_find_K(digitsX, folder_name, plot_prefix[idx])
        else:
            data_analysis_find_K(wineX, folder_name, plot_prefix[idx])
            # Correlation of Wine-Q features
            corr_mat = wine_df.corr()
            corr_matrix = cluster_corr(corr_mat)
            plt.figure(figsize=(10, 10))
            ax = sns.heatmap(corr_matrix, vmin=-1, vmax=1, center=0,
                                cmap=sns.diverging_palette(20, 220, n=200),
                                square=True, annot=True)
            ax.set_xticklabels(ax.get_xticklabels(),
                               rotation=45,
                               horizontalalignment='right')
            plt.title("WineQ Features Correlation Matrix")
            # plt.xticks(rotation=45)
            plt.savefig(folder_name + '{}-'.format(ret_plot_num()) + 'wineQ-FeaturesCorrelation-heatmap.png')
            # plt.show()
            # sns_wq_plot.show()
            plt.close()

            fp.write("Wine-Q data")
            fp.write(str(wine_df.head()))
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            sns.distplot(wine_df.iloc[:, 0], ax=ax[0])
            sns.distplot(wine_df.iloc[:, 1], ax=ax[1])
            sns.distplot(wine_df.iloc[:, 2], ax=ax[2])
            plt.tight_layout()
            plt.savefig(folder_name + '{}-'.format(ret_plot_num()) + 'wineQ-BeforeScaling.png')
            # plt.show()
            plt.close()

            fig1, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].set(xlabel="fixed acidity")
            sns.distplot(x_train[:, 0], ax=ax[0])
            ax[1].set(xlabel="volatile acidity")
            sns.distplot(x_train[:, 1], ax=ax[1])
            ax[2].set(xlabel="citric acid")
            sns.distplot(x_train[:, 2], ax=ax[2])
            plt.tight_layout()
            plt.savefig(folder_name + '{}-'.format(ret_plot_num()) + 'wineQ-AfterScaling.png')
            # plt.show()
            plt.close()

        kmeans_clustering(fp, clusters, s_data, folder_name, plot_prefix[idx], dr="")
        em_clustering(fp, clusters, s_data, folder_name, plot_prefix[idx], dr="")

# Dimensionality reduction
        # selecting how many features to take in PCA
        pca = PCA()  # creates an instance of PCA class
        clf = pca.fit(x_train)  # applies PCA on predictor variables
        plt.title("{}-Scree Plot".format(plot_prefix[idx]))
        plt.plot(clf.explained_variance_ratio_ * 100, marker="o")  # scree plot
        plt.xlabel("num of Features")
        plt.ylabel("Variance Ratio %")
        plt.savefig(folder_name+"{}-{}-Scree Plot".format(ret_plot_num(), plot_prefix[idx]))
        # plt.show()
        plt.close()

        plt.title("{}-PCA Plot".format(plot_prefix[idx]))
        pd.DataFrame(clf.explained_variance_ratio_).plot.bar()
        plt.xlabel("num of Features")
        plt.ylabel("Expected Variance")
        plt.legend('')
        plt.savefig(folder_name + "{}-{}-PCA-Variance Plot".format(ret_plot_num(), plot_prefix[idx]))
        # plt.show()
        plt.close()

        # evaluate the models and store results
        results, names = list(), list()
        n_features = clusters
        # print(n_features)

# 2.1 PCA
        if plot_prefix[idx] == "digits":
            n_features = 20
        else:
            n_features = 6

        clf, ntX, nvX = uld.pca_dim_reduction(s_data, n_features)
        # Plot
        plt.figure(figsize=(10, 6))
        plt.title("Eigenvalue distribution for PCA with " + str(clf.n_components) + " components: " + plot_prefix[idx])
        plt.xlabel('eigenvalue')
        plt.ylabel('frequency')
        plt.xticks(rotation=45)
        if plot_prefix[idx] == "digits":
            minv = -0.01
            maxv = 0.01
        else:
            minv = -0.001
            maxv = 0.1
        bags = np.linspace(minv, maxv, 100)
        for cnt, comp in enumerate(clf.components_):
            plt.hist(comp, bags, alpha=0.5, label=str(cnt + 1))
        plt.legend()
        plt.savefig(folder_name+'{}-{}-'.format(ret_plot_num(), plot_prefix[idx])+"PCA-histogram.png")
        # plt.show()
        plt.close()

# To compare with FA
        n_components = list()
        if plot_prefix[idx] == "digits":
            n_components = np.arange(0, 64, 4)  # options for n_components
        else:
            n_components = np.arange(0, 12, 1)  # options for n_components
        pca_scores = list()
        for n in n_components:
            clf.n_components = n
            pca_scores.append(np.mean(cross_val_score(clf, x_train)))
        index = np.argmax(pca_scores)
        n_components_pca = n_components[index]

# 3.1 Apply new reduced features to Clustering
        new_data = [ntX, nvX, y_train, y_test]
        # Apply new set of data to Clustering algorithms
        kmeans_clustering(fp, n_features, new_data, folder_name, plot_prefix[idx], dr="PCA")
        em_clustering(fp, n_features, new_data, folder_name, plot_prefix[idx], dr="PCA")

# 4&5 - NN
        if plot_prefix[idx] == "wine-Q":
            nn_learner(new_data, folder_name, fp, dr="PCA")

# 2.2 ICA
        clf, ntX, nvX = uld.ica_dim_reduction(s_data, n_features)
        # Plot
        plt.figure(figsize=(10, 6))
        plt.title("Components distribution for ICA with " + str(clf.n_components) + " components: " + plot_prefix[idx])
        plt.xlabel('value')
        plt.ylabel('frequency')
        plt.xticks(rotation=45)
        if plot_prefix[idx] == "digits":
            minv = -0.001
            maxv = 0.001
        else:
            minv = -0.001
            maxv = 0.1
        bags = np.linspace(minv, maxv, 100)
        kurtosis_val = list()
        for cnt, comp in enumerate(clf.components_):
            kurtosis_val.extend(comp)
            lab = str(cnt + 1) + ":" + str(stats.kurtosis(comp))
            plt.hist(comp, bags, alpha=0.5, label=lab)
        plt.legend()
        plt.savefig(folder_name + '{}-{}-'.format(ret_plot_num(), plot_prefix[idx]) + "ICA-histogram.png")
        # plt.show()
        plt.close()
        fp.write("\nICA Kurtosis :")
        fp.write(str(stats.kurtosis(kurtosis_val)))
        fp.write("\n")

# 3.2 Apply new reduced features to Clustering
        new_data = [ntX, nvX, y_train, y_test]
        # Apply new set of data to Clustering algorithms
        kmeans_clustering(fp, n_features, new_data, folder_name, plot_prefix[idx], dr="ICA")
        em_clustering(fp, n_features, new_data, folder_name, plot_prefix[idx], dr="ICA")

# 4&5 - NN
        if plot_prefix[idx] == "wine-Q":
            nn_learner(new_data, folder_name, fp, dr="ICA")

# 2.3 Randomized Projection
        clf, ntX, nvX = uld.rp_dim_reduction(s_data, n_features)
        # Plot
        plt.figure(figsize=(10, 6))
        plt.title("Components distribution for RP with " + str(clf.n_components) + " components: " + plot_prefix[idx])
        plt.xlabel('value')
        plt.ylabel('frequency')
        plt.xticks(rotation=45)
        if plot_prefix[idx] == "digits":
            minv = -1
            maxv = 1
        else:
            minv = -1
            maxv = 1
        bags = np.linspace(minv, maxv, 100)
        kurtosis_val = list()
        for cnt, comp in enumerate(clf.components_):
            kurtosis_val.extend(comp)
            lab = str(cnt + 1) + ":" + str(stats.kurtosis(comp))
            plt.hist(comp, bags, alpha=0.5, label=lab)
        plt.legend()
        plt.savefig(folder_name + '{}-{}-'.format(ret_plot_num(), plot_prefix[idx]) + "RP-histogram.png")
        # plt.show()
        plt.close()
        fp.write("\nRP Kurtosis :")
        fp.write(str(stats.kurtosis(kurtosis_val)))
        fp.write("\n")

# 3.3 Apply new reduced features to Clustering
        new_data = [ntX, nvX, y_train, y_test]
        # Apply new set of data to Clustering algorithms
        kmeans_clustering(fp, n_features, new_data, folder_name, plot_prefix[idx], dr="RP")
        em_clustering(fp, n_features, new_data, folder_name, plot_prefix[idx], dr="RP")

# 4&5 - NN
        if plot_prefix[idx] == "wine-Q":
            nn_learner(new_data, folder_name, fp, dr="RP")

# 2.4 Factor Analysis
        clf, ntX, nvX = uld.fa_dim_reduction(s_data, n_features)
        # Plot
        plt.figure(figsize=(10, 6))
        plt.title("Eigenvalue distribution for FA with " + str(clf.n_components) + " components: " + plot_prefix[idx])
        plt.xlabel('eigenvalue')
        plt.ylabel('frequency')
        plt.xticks(rotation=45)
        if plot_prefix[idx] == "digits":
            minv = -0.01
            maxv = 0.01
        else:
            minv = -0.001
            maxv = 0.1
        bags = np.linspace(minv, maxv, 100)
        for cnt, comp in enumerate(clf.components_):
            plt.hist(comp, bags, alpha=0.5, label=str(cnt + 1))
        plt.legend()
        plt.savefig(folder_name + '{}-{}-'.format(ret_plot_num(), plot_prefix[idx]) + "FA-histogram.png")
        # plt.show()
        plt.close()

# 3.4 Apply new reduced features to Clustering
        new_data = [ntX, nvX, y_train, y_test]
        # Apply new set of data to Clustering algorithms
        kmeans_clustering(fp, n_features, new_data, folder_name, plot_prefix[idx], dr="FA")
        em_clustering(fp, n_features, new_data, folder_name, plot_prefix[idx], dr="FA")

# 4&5 - NN
        if plot_prefix[idx] == "wine-Q":
            nn_learner(new_data, folder_name, fp, dr="FA")

        # To compare with PCA
        fa_scores = list()
        for n in n_components:
            clf.n_components = n
            fa_scores.append(np.mean(cross_val_score(clf, x_train)))
        n_components_fa = n_components[np.argmax(fa_scores)]

        print("best n_components by PCA CV = %d" % n_components_pca)
        print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
        fp.write("best n_components by PCA CV = %d\n" % n_components_pca)
        fp.write("best n_components by FactorAnalysis CV = %d\n" % n_components_fa)
        plt.figure()
        plt.plot(n_components, pca_scores, 'b', label='PCA scores')
        plt.plot(n_components, fa_scores, 'r', label='FA scores')
        plt.axvline(n_components_pca, color='b',
                    label='PCA CV: %d' % n_components_pca, linestyle='--')
        plt.axvline(n_components_fa, color='r',
                    label='FactorAnalysis CV: %d' % n_components_fa,
                    linestyle='--')
        plt.xlabel('number of components')
        plt.ylabel('CV scores')
        plt.legend(loc='lower right')
        plt.title("PCA vs FA Covariance estimation Analysis")
        plt.savefig(folder_name + '{}-'.format(ret_plot_num()) + "PCAvsFA-CE-Analysis.png")
        # plt.show()
        plt.close()

    end = time.time()
    fp.write("\n\n\n")
    exec_time = end-start
    print("\nExecution Time = " + str(exec_time) + "\n")
    fp.write("Start Time = " + str(start) + "\n")
    fp.write("End Time = " + str(end) + "\n")
    fp.write("Execution Time = " + str(exec_time) + "\n")
    fp.write("\n\n\n" + "Done")
    fp.close()
    print("Done")

