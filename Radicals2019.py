"""
Nested sparse Logistic Regression
2019
Author:
        Jeremy Lefort-Besnard  jlefortbesnard (at) tuta (dot) io

"""

import numpy as np
import pandas as pd
from matplotlib import pylab as plt
import seaborn as sns
from scipy.stats import scoreatpercentile
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# reproducibility
np.random.seed(42)

############
### DATA ###
############

df = pd.read_excel("df_full_english.xlsx")
X_colnames = df.columns[2:]

#####################################
### SPARSE LOGISTIC REGRESSSION #####
#####################################

# FITTING MODEL
# hyperparameters & model
n_verticals = 25
p_grid = {"C": np.logspace(-3.5, 1, 25)} # sparsity level
LogReg = LogisticRegression(penalty='l1', verbose=False) # sparsity

# Dealing with class imbalance
df_non_rad = df[df["Condition"]==0]
df_rad = df[df["Condition"]==1]

# upsampling because length [non_rad/rad] = more than 1/3
sample_index_up = np.random.choice(df_non_rad.index, len(df_rad))
df_non_rad_up = df_non_rad.loc[sample_index_up]
df_non_rad_up = df_non_rad_up.reset_index(drop=True)

# merge upsampled dataset
df_merged = pd.concat([df_rad, df_non_rad_up]).reset_index(drop=True)
X = df_merged[df_merged.columns[2:]]
Y = df_merged[df_merged.columns[1]]

# define input and output and standardize the data
y_bin = Y.values
X =  X.values
X_ss = StandardScaler().fit_transform(X)

# nested cross validation, see https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
inner_cv = KFold(n_splits=4, shuffle=True, random_state=42)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=42)

# model assessment
clf = GridSearchCV(estimator=LogReg, param_grid=p_grid, cv=inner_cv,
                       iid=False)

# model evaluation
nested_score = cross_val_score(clf, X=X_ss, y=y_bin, cv=outer_cv)

# Save the best model with its coeficients
coefs = clf.fit(X_ss,y_bin).best_estimator_.coef_
best_estimator = clf.fit(X_ss,y_bin).best_estimator_
best_params = clf.fit(X_ss,y_bin).best_params_
accs = nested_score
acc = np.mean(nested_score)
acc = np.round_(acc * 100, decimals=2, out=None)
print("accs: {}, acc: {}".format(nested_score, np.mean(nested_score)))


# PLOTTING
# Plotting function for the x axis to be centered
def rotateTickLabels(ax, rotation, which, rotation_mode='anchor', ha='left'):
    axes = []
    if which in ['x', 'both']:
        axes.append(ax.xaxis)
    elif which in ['y', 'both']:
        axes.append(ax.yaxis)
    for axis in axes:
        for t in axis.get_ticklabels():
            t.set_horizontalalignment(ha)
            t.set_rotation(rotation)
            t.set_rotation_mode(rotation_mode)

# data = np.round_(coefs, decimals=1, out=None)
data = np.round_(coefs, decimals=2, out=None)

# Do not print 0 in the figure, only coefficients != from 0
labels = []
for i in data[0]: # squeeze
    if i == 0.:
        labels.append("")
    else:
        labels.append(i)
labels = np.array(labels)
labels = labels.reshape((1, len(data[0])))

accuracy_val = "Prediction accuracy = {}%".format(acc)


f, ax = plt.subplots(figsize=(20,8))
sns.heatmap(data, cmap="RdBu_r", vmin=-1.3, vmax=1.3, square=True, annot=labels, fmt = '', annot_kws={"size": 10}, cbar_kws={"shrink": .15})
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_ticklabels(X_colnames, fontsize=14)
rotateTickLabels(ax, 45, 'x')

y_ticks = [" "]
ax.yaxis.set_ticklabels(y_ticks, fontsize=14, rotation=0)
plt.ylabel('Weights', fontsize=16, weight="bold")
plt.text(0.8, -7, accuracy_val, fontsize=14)

plt.tight_layout()
# plt.savefig('SparseLogReg_nested.png', PNG=300)
plt.show()


################################
#### LEARNING CURVE ############
################################

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training sample size", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
title = "Sparse Logistic Regression"
plot_learning_curve(best_estimator, title, X_ss, y_bin, cv=cv, n_jobs=1, ylim=[0.5, 1])
# plt.savefig('Learning_curve.png', PNG=300)
plt.show()


################################
#### CONFUSION MATRIX ##########
################################

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools

# compute the predictions
X_train, X_test, y_train, y_test = train_test_split(X_ss, y_bin, random_state=42)
clf = best_estimator.fit(X_train, y_train)
probs = clf.predict(X_test)

# define confusion matrix input
y_pred = probs
y_true = y_test
f, ax = plt.subplots(figsize=(8, 8))
class_names = ["Radical", "Not radical"]
class_names_fake = ["Radical", "Not radical"]

# plot the matrix
cm = confusion_matrix(y_true, y_pred)
cm_ = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm = (cm_ * 100) # percentage
for indx, i in enumerate(cm):
    for indy, j in enumerate(i):
        j = round(j, 1)
        print(j)
        cm[indx, indy] = j
print(cm)
plt.imshow(cm, vmin=0, vmax=100, interpolation='nearest', cmap=plt.cm.Blues)
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, fontsize=12)
plt.yticks(tick_marks, class_names, fontsize=12)
rotateTickLabels(ax, -55, 'x')
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j]) + "%",
             horizontalalignment="center",
             color= "black", fontsize=15)
plt.xlabel('Predicted label', fontsize=15)
plt.ylabel("True label", fontsize=15)
plt.tight_layout()
# plt.savefig('confusion_matrix.png', PNG=300)
plt.show()


################################
#### COEFFICIENTS BEHAVIOR #####
################################

# rerun the sparse log reg model assessment to get coef at each C
coef_list = []
C_grid = np.logspace(-3.5, 1, 25)
for C in C_grid:
    LogReg = LogisticRegression(penalty='l1', C=C, verbose=False)
    coefs_cgrid = LogReg.fit(X_ss, y_bin).coef_
    coef_list.append(np.round(coefs_cgrid, decimals=1))
coef_list = np.squeeze(np.array(coef_list))


df_coefs = pd.DataFrame(coef_list, index = C_grid, columns = X_colnames)
df_coefs = df_coefs.iloc[:20] # set a limit for the figure, no need to go too far

# Define where our model actually is
best_c_index = np.where(C_grid == best_params['C'])
best_x = C_grid[best_c_index]

# actual plotting
f, axarr = plt.subplots(figsize=(10, 10), facecolor='white')
for ind, column in enumerate(df_coefs.columns):
    axarr.plot(df_coefs.index, df_coefs[column].values, label=column, linewidth=1.5)
axarr.vlines(x= best_x, ymin=-1.5, ymax=1.5, color="grey")
axarr.set_xlabel('0 >> 56 features', fontsize=12)
axarr.set_xticks([])
vline_label = "Best model (acc = {}%)".format(acc)
plt.text(best_x -0.15, 1.51, vline_label , color = "grey")
axarr.legend(fontsize=8.4, bbox_to_anchor=(1.05, 1))
axarr.grid(True)
plt.tight_layout()
# plt.savefig('Coef_beh.png, PNG=300')
plt.show()



############################################
###### Bootstrap confidence intervals ######
############################################

# compute error bars:
coef_bs = []
for i_subsample in range(1000):
    # make bootstrap sample
    sample_index = np.random.choice(X_ss.shape[0], len(X_ss))
    X_bs = X_ss[sample_index]
    y_bs = y_bin[sample_index]

    # run Sparse Log Reg in the BS sample
    clf = best_estimator
    clf.fit(X_bs, y_bs)
    acc_bs = clf.score(X_bs, y_bs)

    # collect coefs of the subsample
    coef_bs.append(clf.coef_[0])
    print("acc: {}, bs: {}".format(acc_bs, i_subsample))

# gather error bars and adapt the format for plotting:
df_bs = pd.DataFrame(np.array(coef_bs), columns = X_colnames)
bs_err_dic = {}
for ind, item in enumerate(df_bs):
    bs_err = scoreatpercentile(df_bs[item], [5, 95], interpolation_method='fraction', axis=None)
    bs_err_dic[item] = bs_err

# calculating boostrap CI + and - values above and under the original data
coef_original = np.squeeze(coefs)
err_l_ = []
err_u_ = []
for ind, name in enumerate(X_colnames):
    err_l = coef_original[ind] - bs_err_dic[name][0]
    err_u = bs_err_dic[name][1] - coef_original[ind]
    err_l_.append(err_l)
    err_u_.append(err_u)
err = np.array([err_l_, err_u_])


# plotting the results
X = X_colnames
coef_original_ = np.round_(coef_original, decimals=2, out=None)
Y = coef_original_ # the weights obtained with the original sample

# actual plotting
fig, ax = plt.subplots(figsize=(15,9))
ax.errorbar(X, Y, yerr = err, fmt='mo', ecolor="c", ms=4)
# customize the figure
plt.text(3, -1.7, "Acc=%0.2f, n=%i" %(acc, len(X_ss)), fontsize=12, ha='center', style="italic")
ax.set_xticklabels(X)
ax.xaxis.set_ticks_position('top')
# rotateTickLabels(ax, -55, 'x')
rotateTickLabels(ax, 45, 'x')
plt.xticks(fontsize=12)
plt.xlabel("")
plt.ylabel("Weight", fontsize=12, weight="bold")
plt.yticks(fontsize=12)
plt.axhline(0, color='black')
# plt.grid(True, axis="y")
plt.grid(True, axis="x")
plt.ylim([-2, 2.05])
plt.text(47.5, 1.2, "Radical", fontsize=12, rotation=-90)
plt.text(47.5, -0.6, "Non radical", fontsize=12, rotation=-90)
# plt.suptitle("Boostrap confidence intervals", y=1., fontsize=16, weight="bold")
plt.tight_layout()
# plt.savefig("BootstrapCI_nested.png", PNG=300)
plt.show()
