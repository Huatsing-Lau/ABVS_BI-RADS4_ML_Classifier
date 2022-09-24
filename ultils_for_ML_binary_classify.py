# -*- coding: utf-8 -*-
# +
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split # 分割数据模块
from sklearn import metrics
import copy

seed = 42
import random
random.seed( seed )
import numpy as np
np.random.seed(seed)
# -

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:37:14 2019
采用FCN方法做数据挖掘的分析
p次k折交叉验证
决策边界可视化
参考：
[1] https://www.jianshu.com/p/97557b463df8
@author: liuhuaqing
"""
import umap
import matplotlib.pyplot as plt 
# import tensorflow.compat.v1 as tf
import tensorflow as tf #tensorflow version should be > 2
import tensorflow.keras as keras

import numpy as np
from sklearn.model_selection import cross_validate

# +
try:
    from sklearn.neighbors.classification import KNeighborsClassifier
except:
    from sklearn.neighbors import KNeighborsClassifier
try:
    from sklearn.manifold.t_sne import TSNE
except:
    from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split # 分割数据模块
from sklearn.metrics import confusion_matrix
# -

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# +
from sklearn.utils.validation import _deprecate_positional_args
# @_deprecate_positional_args
def plot_roc_curve(estimator, X, y, *, sample_weight=None,
                   drop_intermediate=True, response_method="auto",
                   name=None, ax=None, **kwargs):
    from sklearn.metrics import auc
    from sklearn.metrics import roc_curve, RocCurveDisplay
    
    from sklearn.utils import check_matplotlib_support
    from sklearn.base import is_classifier   
    
    check_matplotlib_support('plot_roc_curve')

    classification_error = (
        "{} should be a binary classifier".format(estimator.__class__.__name__)
    )
    if not is_classifier(estimator):
        raise ValueError(classification_error)
    
#     from sklearn.metrics._plot.base import _check_classifer_response_method
#     prediction_method = _check_classifer_response_method( estimator, response_method )
#     y_pred = prediction_method(X)
    
    y_pred = estimator.predict_proba(X)

    if y_pred.ndim != 1:
        if y_pred.shape[1] != 2:
            raise ValueError(classification_error)
        else:
            y_pred = y_pred[:, 1]

    pos_label = estimator.classes_[1]
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=pos_label,
                            sample_weight=sample_weight,
                            drop_intermediate=drop_intermediate)
    roc_auc = auc(fpr, tpr)
    name = estimator.__class__.__name__ if name is None else name
    viz = RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name
    )
    viz.threshold = threshold
    return viz.plot(ax=ax, name=name, **kwargs)


# -

# 画混淆矩阵
import matplotlib.pyplot as plt
import seaborn as sns
def plot_confusion_matrix(
    cm,class_names,
    title='Confusion Matrix',
    fn='Confusion Matrix.png',
    fmt='.2g',
    center=None):
    '''
    example:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true = truelabel, y_pred = predClasses)
    sns.set()
    plot_cm(cm,class_names,
            title='Confusion Matrix Model',
            fn='Confusion Matrix Model.png',
            fmt='.20g',
            center=cm.sum()/num_classes
           )
    '''
    f,ax = plt.subplots(dpi=60)
    ax = sns.heatmap(cm,annot=True,fmt=fmt,center=center,annot_kws={'size':20,'ha':'center','va':'center'})#fmt='.20g',center=250
    ax.set_title(title,fontsize=20)#图片标题文本和字体大小
    ax.set_xlabel('Predict',fontsize=20)#x轴label的文本和字体大小
    ax.set_ylabel('Ground-Truth',fontsize=20)#y轴label的文本和字体大小
    ax.set_xticklabels(class_names,fontsize=20)#x轴刻度的文本和字体大小
    ax.set_yticklabels(class_names,fontsize=20)#y轴刻度的文本和字体大小
    #设置colorbar的刻度字体大小
    cax = plt.gcf().axes[-1]
    cax.tick_params(labelsize=20)
    
    if fn:
        plt.savefig(fn, bbox_inches='tight')
    plt.show()
    return f,ax

def plot_3cm(cm,class_names,num_classes,clf_name,dir_result):
    plot_confusion_matrix(cm,class_names,
            title='Confusion Matrix\n'+clf_name,
            fn=os.path.join(dir_result,'Confusion Matrix '+clf_name+'.png'),
            fmt='.20g',
            center=cm.sum()/num_classes
           )
    
    cm_norm_recall = cm.astype('float') / cm.sum(axis=0) 
    cm_norm_recall = np.around(cm_norm_recall, decimals=3)
    plot_confusion_matrix(cm_norm_recall,class_names,
            title='Row-Normalized Confusion Matrix\n'+clf_name,
            fn=os.path.join(dir_result,'Row-Normalized Confusion Matrix '+clf_name+'.png'),
            fmt='.3g',
            center=0.5
           )
    
    cm_norm_precision = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  
    cm_norm_precision = np.around(cm_norm_precision, decimals=3)
    plot_confusion_matrix(cm_norm_precision,class_names,
            title='Column-Normalized Confusion Matrix\n'+clf_name,
            fn=os.path.join(dir_result,'Column-Normalized Confusion Matrix '+clf_name+'.png'),
            fmt='.3g',
            center=0.5
           )


def run_RepeatedKFold(n_splits,n_repeats,classifier,X,Y,title,dir_result):
#    from sklearn.metrics import plot_roc_curve, auc
    from sklearn.metrics import auc
    from sklearn.model_selection import RepeatedKFold
    fold = RepeatedKFold(n_splits=n_splits,n_repeats=n_repeats)
    
    tprs = []
    aucs = []
    thresholds = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(dpi=100)
    for i,(train_index, test_index) in enumerate(fold.split(Y)):
        if X.__class__ == pd.core.frame.DataFrame:
            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]
        elif X.__class__ == np.ndarray:
            X_train = X[train_index]
            X_test = X[test_index]
        Y_train = Y[train_index]
        Y_test = Y[test_index]
        
        classifier.fit(X_train,Y_train)
        viz = plot_roc_curve(classifier, X_test, Y_test,
                         label=None,    
                         #name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        interp_threshold = np.interp(mean_fpr, viz.fpr, viz.threshold)
        interp_threshold[0] = 1.0
        thresholds.append(interp_threshold)
        aucs.append(viz.roc_auc)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    mean_threshold = np.mean(thresholds, axis=0)
    mean_threshold[-1] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title=title)
    ax.legend(loc="lower right")
    plt.savefig( os.path.join(dir_result,title.replace('\n','')) )
    plt.show()
    ret=dict({})   
    ret['mean_fpr'],ret['mean_tpr'],ret['mean_threshold'] = mean_fpr, mean_tpr, mean_threshold
    ret['mean_auc'],ret['aucs'] = mean_auc, aucs
    return ret

def visualize_data_reduced_dimension(data,reducer='UMAP',
                                     n_dim = 3,
                                     title="Reduced Dimension Projection",
                                     dir_result='./'):
    '''
    数据降维可视化
        data:字典，包含：
            X：输入特征
            Y:真实类别
    '''
    class_names = data['class_names']#类别名称
    if reducer == 'TNSE':
        reducer = TSNE(n_components = n_dim, random_state=42)
        X_embedding = reducer.fit_transform(data['X'])
    elif reducer == 'UMAP':
        reducer = umap.UMAP(n_components=n_dim, random_state=42)
        X_embedding = reducer.fit_transform(data['X']) 
        
    X2d_xmin,X2d_xmax = np.min(X_embedding[:,0]), np.max(X_embedding[:,0])
    X2d_ymin,X2d_ymax = np.min(X_embedding[:,1]), np.max(X_embedding[:,1])
    if n_dim==3:
        X2d_zmin,X2d_zmax = np.min(X_embedding[:,2]), np.max(X_embedding[:,2])
        
    #plot 
    fig = plt.figure(dpi=100)#figsize=(8,6))
    colors = ['red','green','blue','orange','cyan']
    markers = ['o','s','^','*']

    if n_dim == 2:
        for i in range(len(class_names)):
            idx = np.where(data['Y']==i)[0].tolist()#根据真实类别标签来绘制散点
            plt.scatter(X_embedding[idx, 0], X_embedding[idx, 1], 
                        c=colors[i],# cmap='Spectral', 
                        marker=markers[i],s=5,
                        label=class_names[i],
                        alpha=0.5, 
                        edgecolors='none'
                       )
    elif n_dim == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel('dim 3')
#         plt.zlim(X2d_zmin,X2d_zmax)
        for i in range(len(class_names)):
            idx = np.where(data['Y']==i)[0].tolist()#根据真实类别标签来绘制散点
            ax.scatter(X_embedding[idx, 0].squeeze(), 
                       X_embedding[idx, 1].squeeze(), 
                       X_embedding[idx, 2].squeeze(),
                       c=colors[i],
                       s=5,
                       marker=markers[i],
                       label=class_names[i],
                       alpha=0.5, 
                      )
                 
    ax.set_xlabel('dim 1')
    ax.set_ylabel('dim 2')  
#     plt.xlim(X2d_xmin,X2d_xmax)
#     plt.ylim(X2d_ymin,X2d_ymax)       
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    fn = os.path.join(dir_result,title.replace('/n','_')+".png")
    plt.savefig( fn )
    plt.show()
    
    return fig


def visualize_reduced_decision_boundary(clf,data,reducer='UMAP',
                                        title="Decision-Boundary of the Trained Classifier in Reduced-Features-Space",
                                       dir_result='./'):
    '''
    分类器的决策边界可视化
    clf:分类器
    X：输入特征
    Y:真实类别
    '''
    if reducer == 'TNSE':
        reducer = TSNE(n_components = 2, random_state=42)
        X_embedding = reducer.fit_transform(data['X'])
    elif reducer == 'UMAP':
        reducer = umap.UMAP(n_components=2, random_state=42)
        X_embedding = reducer.fit_transform(data['X']) 
    
    y_predicted = clf.predict(data['X'])#根据y_predicted 结合KNeighborsClassifier来确定决策边界
    
    # 生成新的数据, 并调用meshgrid网格搜索函数帮助我们生成矩阵
    # 创建meshgrid 
    resolution = 400 #100x100背景像素
    X2d_xmin,X2d_xmax = np.min(X_embedding[:,0]), np.max(X_embedding[:,0])
    X2d_ymin,X2d_ymax = np.min(X_embedding[:,1]), np.max(X_embedding[:,1])
    xx,yy = np.meshgrid(np.linspace(X2d_xmin,X2d_xmax,resolution), np.linspace(X2d_ymin,X2d_ymax,resolution))
     
    #使用1-NN 
    #在分辨率x分辨率网格上近似Voronoi镶嵌化
    background_model = KNeighborsClassifier(n_neighbors = 1).fit(X_embedding,y_predicted)
    voronoiBackground = background_model.predict( np.c_[xx.ravel(),yy.ravel()] )
    voronoiBackground = voronoiBackground.reshape((resolution,resolution))
     
    #plot 
    plt.figure(dpi=100)#figsize=(8,6))
    plt.contourf(xx,yy,voronoiBackground,alpha=0.2)
    idx_0 = np.where(data['Y']==0)[0].tolist()#根据真实类别标签来绘制散点
    idx_1 = np.where(data['Y']==1)[0].tolist()#根据真实类别标签来绘制散点
    plt.scatter(X_embedding[idx_0, 0], X_embedding[idx_0, 1], 
                c='blue',# cmap='Spectral', 
                marker='o',s=20,
                label=data['classname'][0],
                )
    plt.scatter(X_embedding[idx_1, 0], X_embedding[idx_1, 1], 
                c='orange',# cmap='Spectral', 
                marker='s',s=20,
                label=data['classname'][1],
                )
    plt.legend()
    plt.xlim(X2d_xmin,X2d_xmax)
    plt.ylim(X2d_ymin,X2d_ymax)
    #plt.gca().set_aspect('equal', 'datalim')
    plt.title(title)
    plt.savefig( os.path.join(dir_result,title.replace('/n','_')+".png") )
    plt.show()

def UMAP_visualize_decision_boundary(clf,data):
    '''
    分类器的决策边界可视化
    clf:分类器
    X：输入特征
    Y:真实类别
    '''
    reducer = umap.UMAP(n_components=2,random_state=42)
    X_embedding = reducer.fit_transform(data['X'])
    
    # 生成新的数据, 并调用meshgrid网格搜索函数帮助我们生成矩阵
    #创建meshgrid 
    resolution = 100 #100x100背景像素
    X2d_xmin,X2d_xmax = np.min(X_embedding[:,0]), np.max(X_embedding[:,0])
    X2d_ymin,X2d_ymax = np.min(X_embedding[:,1]), np.max(X_embedding[:,1])
    xx,yy = np.meshgrid(np.linspace(X2d_xmin,X2d_xmax,resolution), np.linspace(X2d_ymin,X2d_ymax,resolution))
     
     
    # 有了新的数据, 我们需要将这些数据输入到分类器获取预测结果
    Z = clf.predict( reducer.inverse_transform(np.c_[xx.ravel(), yy.ravel()]) )
    # 这个时候得到的是Z还是一个向量, 将这个向量转为矩阵即可
    Z = Z.reshape(xx.shape)
    plt.figure()
    # 分解的时候有背景颜色    
    plt.figure(figsize=(8,6))
    #plt.pcolormesh(xx, yy, Z, cmap=plt.cm.RdYlBu)
    plt.contourf(xx, yy, Z, alpha=0.2)
    idx_0 = np.where(data['Y']==0)[0].tolist()#根据真实类别标签来绘制散点
    idx_1 = np.where(data['Y']==1)[0].tolist()
    plt.scatter(X_embedding[idx_0, 0], X_embedding[idx_0, 1], 
                c='blue',# cmap='Spectral', 
                marker='o',
                label='benign',
                s=20)
    plt.scatter(X_embedding[idx_1, 0], X_embedding[idx_1, 1], 
                c='orange',# cmap='Spectral', 
                marker='s',
                label='malignant',
                s=20
                )
    plt.legend()
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('Decision-Boundary of the Classifier in Reduced Dimension Projection')
    plt.savefig('Decision-Boundary of the Classifier in Reduced Dimension Projection.png')
    plt.show()
    return None


def plot_DCA(y_proba,y_true,fn="DCA.png"):
    """
    DCA曲线
    """
    pt_arr = []
    net_bnf_arr = []
    y_proba = y_proba.ravel()
    for i in range(0,100,1):
        pt = i /100
        #compute TP FP
        y_proba_clip = (y_proba>pt).astype(int)
        TP = np.sum( y_true*np.round(y_proba_clip) )
        FP = np.sum((1 - y_true) * np.round(y_proba_clip))
        net_bnf = ( TP-(FP * pt/(1-pt)) )/y_true.shape[0]
        pt_arr.append(pt)
        net_bnf_arr.append(net_bnf)
    fig = plt.figure()
    plt.plot(pt_arr, net_bnf_arr, color='red', lw=2, linestyle='-',label='test')
    plt.plot(pt_arr, np.zeros(len(pt_arr)), color='k', lw=1, linestyle='--',label='All negative')
    pt_np = np.array(pt_arr)
    pi = np.sum(y_true)/len(y_true)# 患病率
    all_pos = pi-(1-pi)*pt_np/(1-pt_np)
    plt.plot(pt_arr, all_pos , color='b', lw=1, linestyle='dotted',label='All positive')
    plt.xlim([0.0, 1.0])
    plt.ylim([min(-0.15,min(net_bnf_arr)*1.2), max(0.4,max(net_bnf_arr)*1.2)])
    plt.xlabel('Probability Threshold')
    plt.ylabel('Net Benefit')
    plt.title('DCA')
    plt.legend()
    plt.grid("on")
    plt.savefig(fn)
    plt.show()
    return fig

# 1. Filter方法(逐个特征分析，没有考虑到特征之间的关联作用，可能把有用的关联特征误踢掉。)
#     1.1 移除低方差的特征 (Removing features with low variance)
#     1.2 单变量特征选择 (Univariate feature selection)
#         1.2.1 卡方(Chi2)检验
#         1.2.2 互信息和最大信息系数 Mutual information and maximal information coefficient (MIC)
#         1.2.3 基于模型的特征排序 (Model based ranking)
# 2. Wrapper
#     2.1 RFE
# 3.Embedding
#     3.1 使用SelectFromModel选择特征 (Feature selection using SelectFromModel)
#         3.1.1 基于L1的特征选择 (L1-based feature selection)


def plt_feature_importance(feat_imp,dir_result='./',fig=None):
    # 画特征重要性bar图
    y_pos = np.arange(feat_imp.shape[0])
    plt.barh(y_pos, feat_imp.values.squeeze(), align='center', alpha=0.8)
    plt.yticks(y_pos, feat_imp.index.values)
    plt.xlabel('Feature Importance')
    plt.title('Feature Selection')
    fn = os.path.join(dir_result,'Feature Importance.png')
    plt.savefig(fn, bbox_inches='tight')
    plt.show()
    return fig

# Filter
def select_features_Chi2(X,Y,kbest=10,dir_result='./'):
    """
    采用卡方检验(Chi2)方法(SelectKBest)选择特征
    注意：经典的卡方检验是检验定性自变量对定性因变量的相关性。
    注意：Input X must be non-negative.
    """
    from sklearn.feature_selection import SelectKBest, chi2
    fit = SelectKBest(score_func=chi2, k=kbest).fit(X, Y)
    index_selected_features = np.where(fit.support_==True)[0].tolist()
    return index_selected_features

def selelct_features_MIC(X,Y,kbest=15,dir_result='./'):
    """采用互信息指标，来筛选特征"""
    from minepy import MINE#minepy包——基于最大信息的非参数估计
    m = MINE()
    # 单独采用每个特征进行建模
    feature_names = X.columns
    importance = []
    for i in range(X.shape[1]):
        m.compute_score( X.iloc[:, i], Y)
        importance.append( round(m.mic(),3) )
    # 重新排序
    feat_imp = pd.DataFrame(data=np.array([importance]).T, index=feature_names, columns=['feature importance'])
    feat_imp.sort_values(by=['feature importance'],axis='index', ascending=False,inplace=True)
    feat_imp = feat_imp.iloc[:kbest]
    
    # 画图
    fig = plt.figure(dpi=100,figsize=(6.4, 4.8*1.5))
    fig = plt_feature_importance(feat_imp,dir_result,fig)

    return feat_imp


# Wrapper
def select_features_RFE(X,Y,base_model,kbest=15,dir_result='./'):
    """采用RFE方法选择特征，用户可以指定base_model"""
    from sklearn.feature_selection import RFE
    rfe = RFE(base_model, n_features_to_select=kbest)
    fit = rfe.fit(X, Y)
    
    feature_names = X.columns.values.tolist()
    #　挑选出的特征
    index_selected_features = np.where(fit.support_==True)[0].tolist()
    importance = fit.estimator_.coef_.squeeze().tolist()
    #importance = [fit.ranking_[k] for k in index_selected_features]
    feature_names = [feature_names[k] for k in index_selected_features]
    # 重新排序
    feat_imp = pd.DataFrame(data=np.array([importance]).T, index=feature_names, columns=['feature importance'])
    feat_imp.sort_values(by=['feature importance'],axis='index', ascending=False,inplace=True)
    
    # 画图
    fig = plt.figure(dpi=100)#,figsize=(6.4, 4.8*1.5))
    fig = plt_feature_importance(feat_imp,dir_result,fig)
        
    return feat_imp

# Embedding
def select_features_LSVC(X,Y,dir_result='./'):
    """采用LSVC方法选择特征""" 
    from sklearn.feature_selection import SelectFromModel
    from sklearn.svm import LinearSVC

    my_model = LinearSVC(C=0.01, penalty="l1", dual=False).fit( X, Y )
    selector = SelectFromModel(my_model,prefit=True)
    importance = selector.estimator.coef_.squeeze().tolist()
    feature_names = X.columns.values.tolist()
    #　挑选出的特征
    index_selected_features = np.where(selector.get_support()==True)[0].tolist()
    importance = [importance[k] for k in index_selected_features]
    feature_names = [feature_names[k] for k in index_selected_features]
    # 重新排序
    feat_imp = pd.DataFrame(data=np.array([importance]).T, index=feature_names, columns=['feature importance'])
    feat_imp.sort_values(by=['feature importance'],axis='index', ascending=False,inplace=True)
    
    # 画图
    fig = plt.figure(dpi=100,figsize=(6.4, 4.8*1.5))
    fig = plt_feature_importance(feat_imp,dir_result,fig)
    
    return feat_imp   

def select_features_LR(X,Y,dir_result='./'):
    """采用带L1和L2惩罚项的逻辑回归作为基模型的特征选择,
    参数threshold为权值系数之差的阈值""" 
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression as LR

    my_model = LR(C=0.1).fit( X, Y )
    selector = SelectFromModel(my_model,prefit=True)
    importance = selector.estimator.coef_.squeeze().tolist()
    feature_names = X.columns.values.tolist()
    #　挑选出的特征
    index_selected_features = np.where(selector.get_support()==True)[0].tolist()
    importance = [importance[k] for k in index_selected_features]
    feature_names = [feature_names[k] for k in index_selected_features]
    # 重新排序
    feat_imp = pd.DataFrame(data=np.array([importance]).T, index=feature_names, columns=['feature importance'])
    feat_imp.sort_values(by=['feature importance'],axis='index', ascending=False,inplace=True)

    # 画图
    fig = plt.figure(dpi=100,figsize=(6.4, 4.8*1.5))
    fig = plt_feature_importance(feat_imp,dir_result,fig)
        
    return feat_imp

def select_features_Tree(X,Y,dir_result='./'):
    """采用Tree的方法选择特征""" 
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import ExtraTreesClassifier
#     selector = SelectFromModel(ExtraTreesClassifier()).fit(X, Y)
#     index_selected_features = np.where(selector.get_support()==True)[0].tolist()
    
    my_model = ExtraTreesClassifier().fit( X, Y )
    selector = SelectFromModel(my_model,prefit=True)
    importance = selector.estimator.feature_importances_.squeeze().tolist()
    feature_names = X.columns.values.tolist()
    #　挑选出的特征
    index_selected_features = np.where(selector.get_support()==True)[0].tolist()
    importance = [importance[k] for k in index_selected_features]
    feature_names = [feature_names[k] for k in index_selected_features]
    # 重新排序
    feat_imp = pd.DataFrame(data=np.array([importance]).T, index=feature_names, columns=['feature importance'])
    feat_imp.sort_values(by=['feature importance'],axis='index', ascending=False,inplace=True)

    # 画图
    fig = plt.figure(dpi=100,figsize=(6.4, 4.8*1.5))
    fig = plt_feature_importance(feat_imp,dir_result,fig)
    
    return feat_imp

def select_features_RF(X,Y,dir_result='./'):
    """基于模型（此处采用随机森林交叉验证）的特征排序，"""
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestRegressor
    RF = RandomForestRegressor(n_estimators=20, max_depth=4)
    
    my_model = RF.fit( X, Y )
    selector = SelectFromModel(my_model,prefit=True)
    importance = selector.estimator.feature_importances_.squeeze().tolist()
    feature_names = X.columns.values.tolist()
    #　挑选出的特征
    index_selected_features = np.where(selector.get_support()==True)[0].tolist()
    importance = [importance[k] for k in index_selected_features]
    feature_names = [feature_names[k] for k in index_selected_features]
    # 重新排序
    feat_imp = pd.DataFrame(data=np.array([importance]).T, index=feature_names, columns=['feature importance'])
    feat_imp.sort_values(by=['feature importance'],axis='index', ascending=False,inplace=True)

    # 画图
    fig = plt.figure(dpi=100,figsize=(6.4, 4.8*1.5))
    fig = plt_feature_importance(feat_imp,dir_result,fig)
    return feat_imp

def select_features_mrmr(X,method='MID',kbest=10, dir_result='./'):
    """
    采用mRMR方法筛选特征(该方法不考虑应变量)
    X是dataframe
    """
    import pymrmr
    name_selected_features = pymrmr.mRMR(X, method, kbest)#也可以输入dataframe
    feat_imp = pd.DataFrame( data=np.zeros([kbest,1]), index=name_selected_features, columns=['feature importance'])
    return feat_imp


def select_features(X,Y,class_names,method,dir_result,**kwargs):
    """多种特征选择方法的封装函数"""
    if 'kbest' in kwargs.keys():
        kbest = kwargs['kbest']
    else:
        kbest = 20


    if method == 'MIC':
        feat_imp = selelct_features_MIC(X,Y,kbest=kbest,dir_result=dir_result)
    elif method == 'RFE':
        from sklearn.linear_model import LogisticRegression as LR
        LogR = LR(penalty='l2',dual=False,tol=0.0001,C=1.0,fit_intercept=True,
           intercept_scaling=1,class_weight=None,random_state=None,
           solver='liblinear',max_iter=1000,multi_class='ovr',
           verbose=0,warm_start=False,n_jobs=1)
        feat_imp = select_features_RFE(X,Y,class_names,base_model=LogR,kbest=kbest,dir_result=dir_result)
    elif method == 'EmbeddingLSVC':
        feat_imp = select_features_LSVC(X,Y,class_names,dir_result)
    elif method == 'EmbeddingLR':
        feat_imp = select_features_LR(X,Y,class_names,dir_result)
    elif method == 'EmbeddingTree':
        feat_imp = select_features_Tree(X,Y,class_names,dir_result)
    elif method == 'EmbeddingRF':
        feat_imp = select_features_RF(X,Y,class_names,dir_result)
    elif method == 'mRMR':
        feat_imp = select_features_mrmr(X,'MID',kbest=kbest,dir_result=dir_result)

    if feat_imp.__class__ == list:
        Data = {'X':X.loc[:,feat_imp[0].index],'Y':Y}
    elif feat_imp.__class__ == pd.core.frame.DataFrame:
        Data = {'X':X.loc[:,feat_imp.index],'Y':Y}
    return Data,feat_imp


def select_features(X,Y,method,dir_result,**kwargs):
    """多种特征选择方法的封装函数"""
    if 'kbest' in kwargs.keys():
        kbest = kwargs['kbest']
    else:
        kbest = 20

    if method == 'MIC':
        feat_imp = selelct_features_MIC(X,Y,kbest=kbest,dir_result=dir_result)
    elif method == 'RFE':
        from sklearn.linear_model import LogisticRegression as LR
        LogR = LR(penalty='l2',dual=False,tol=0.0001,C=1.0,fit_intercept=True,
           intercept_scaling=1,class_weight=None,random_state=None,
           solver='liblinear',max_iter=1000,multi_class='ovr',
           verbose=0,warm_start=False,n_jobs=1)
        feat_imp = select_features_RFE(X,Y,base_model=LogR,kbest=kbest,dir_result=dir_result)
    elif method == 'EmbeddingLSVC':
        feat_imp = select_features_LSVC(X,Y,dir_result)
    elif method == 'EmbeddingLR':
        feat_imp = select_features_LR(X,Y,dir_result)
    elif method == 'EmbeddingTree':
        feat_imp = select_features_Tree(X,Y,dir_result)
    elif method == 'EmbeddingRF':
        feat_imp = select_features_RF(X,Y, dir_result=dir_result)
    elif method == 'mRMR':
        feat_imp = select_features_mrmr(X,'MID',kbest=kbest, dir_result=dir_result)

    Data = {'X':X.loc[:,feat_imp.index],'Y':Y}
    return Data,feat_imp



def boxplot(x,x_names,title='AUC of different Algorithms',fn_save='AUC-of-different-Algorithms.png'):
    # 参考：https://blog.csdn.net/roguesir/article/details/78249864 
    fig=plt.figure(dpi=100)
    plt.boxplot(x,patch_artist=True, # 要求用自定义颜色填充盒形图，默认白色填充
                showmeans=True, # 以点的形式显示均值 
                boxprops = {'color':'black','facecolor':'#9999ff'}, # 设置箱体属性，填充色和边框色           
                flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}, # 设置异常值属性，点的形状、填充色和边框色
                meanprops = {'marker':'D','markerfacecolor':'indianred'}, # 设置均值点的属性，点的形状、填充色 
                medianprops = {'linestyle':'--','color':'orange'}) # 设置中位数线的属性，线的类型和颜色
    plt.xticks([y+1 for y in range(len(x))], x_names, rotation='vertical')
    plt.ylabel('Cross Validation AUC')
    plt.title(title)
    plt.savefig(fn_save, bbox_inches='tight')
    plt.show() 
    return fig

def get_algorithm_external_result(
    X,Y,
    classifier_fitted,
    clf_name, 
    dir_result='./'
):
    """
    外部验证，输出roc曲线、混淆矩阵
    classifier_fitted
    """
    title='ROC Curve of '+clf_name+' on External Data-set\n'
    dir_result = os.path.join(dir_result,'external-dataset_'+clf_name)
    if not os.path.isdir(dir_result):
        os.mkdir(dir_result)
     
    # roc曲线
    fig, ax = plt.subplots(dpi=100)
    viz = plot_roc_curve(classifier_fitted, X, Y,
                         label=None,#'external dataset',  
                         name=Ｎone,#'ROC fold {}'.format(i),
                         ax=ax)

    roc_auc = viz.roc_auc
#     viz.ax_.set_label(r'ROC (AUC = %0.3f)' % (roc_auc))  
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
    ax.legend([r'ROC (AUC = %0.3f)' % (roc_auc),'Chance'],loc="lower right")
    plt.savefig( os.path.join(dir_result,title.replace('\n','')) )
    plt.show()
    
    # 画混淆矩阵
    y_pred = np.array(classifier_fitted.predict(X),dtype=np.int)
    cm = confusion_matrix( y_true = Y, y_pred = y_pred )
    class_names = ['<=7 d','>7 d']
    num_classes = 2
    plot_3cm(cm,class_names,num_classes,clf_name,dir_result)

#     # 画决策边界
#     data = dict(X=X,Y=Y,classname=['no','yes'])
#     visualize_reduced_decision_boundary(
#         clf=classifier_fitted,data=data,
#         title="Decision-Boundary of "+clf_name+" in TNSE-Projection",
#         dir_result=dir_result
#     ) 
    ret = dict(auc=roc_auc,cm=cm)
    return ret


# +
def get_algorithm_test_result(
    X,Y,class_names,
    classifier_fitted,
    best_threshold=None,
    clf_name=None,
    dataset_name=None,
    dir_result='./'
):
    """
    外部验证，输出roc曲线、混淆矩阵
    classifier_fitted
    """
    
    dir_result = os.path.join(dir_result,dataset_name)
    if not os.path.isdir(dir_result):
        os.mkdir(dir_result)
    
    # (1)测试集上的ROC曲线及其AUC
    fig, ax = plt.subplots(dpi=100)
    title='ROC Curve of '+clf_name+' on '+dataset_name
    fn =  os.path.join(dir_result,title.replace('\n','_'))
    viz = plot_roc_curve(classifier_fitted, X, Y,
                         label=None,    
                         name=None,
                         ax=ax)
    roc_auc = viz.roc_auc
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
    ax.legend([r'ROC (AUC = %0.3f)' % (roc_auc),'Chance'],loc="lower right")
    plt.savefig( fn )
    plt.show()
        
    # (2)画混淆矩阵
    if best_threshold:
        y_pred = np.argmax( np.array(classifier_fitted.predict_proba(X)>best_threshold, dtype=np.int), axis=1 )
    else:
        y_pred = classifier_fitted.predict(X)
    
    cm = confusion_matrix( y_true = Y, y_pred = y_pred )
    num_classes = len(class_names)
    plot_3cm(cm,class_names,num_classes,clf_name,dir_result)
    
    # (3)获取report： specificity,precision,recall,f1-score等
    # 1)获取原始report
    from sklearn.metrics import classification_report
    report = classification_report(
        y_true=Y, 
        y_pred=y_pred, 
        labels=np.unique(Y).tolist(),
        target_names=class_names,
        digits=3,
        output_dict=False
    )
    ## 2)补充roc_auc
    report = report+'\n'+'roc_auc: '+str(roc_auc.round(decimals=3))
    ## 3)保存
    fn = os.path.join(dir_result,'classification report of '+clf_name+' against test set.txt')
    save_text(fn, report)
    print(report)
    
    # (4)DCA
    fn = os.path.join(dir_result,"DCA.png")
    plot_DCA(y_proba=classifier_fitted.predict_proba(X)[:,1], y_true=Y, fn=fn)

#     # 画决策边界
#     data = dict(X=X,Y=Y,classname=['no','yes'])
#     visualize_reduced_decision_boundary(
#         clf=classifier_fitted,data=data,
#         title="Decision-Boundary of "+clf_name+" in TNSE-Projection",
#         dir_result=dir_result
#     ) 
    ret = dict(roc_auc=roc_auc,cm=cm,report=report)
    return ret


# -

from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
def plot_calibration_curve_binary_class(
    clf, 
    clf_name, 
    X_train, y_train,
    X_test, y_test,
    cv='prefit',#5
    n_bins = 10,
    dir_result='./'
):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with sigmoid calibration
    calibrated_classifier = CalibratedClassifierCV(clf, cv=cv, method='sigmoid')


    fig = plt.figure(dpi=100,figsize=(5, 5))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    
    for est, name in [(clf, clf_name),
                      (calibrated_classifier, "calibrated "+clf_name)]:
        est.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(est, "predict_proba"):
            prob_pos = est.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = est.decision_function(X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        est_score = brier_score_loss(y_test, prob_pos, pos_label=y_test.max())

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=n_bins)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, est_score))

        ax2.hist(prob_pos, range=(0, 1), bins=n_bins, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives",fontsize=10)
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)',fontsize=10)

    ax2.set_xlabel("Mean predicted value",fontsize=10)
    ax2.set_ylabel("Count",fontsize=10)
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    
    fn = os.path.join(dir_result,'Calibration plots.png')
    plt.savefig( fn )
    plt.show()
    return fig,calibrated_classifier

def save_text(filename, contents):
    fh = open(filename, 'w', encoding='utf-8')
    fh.write(contents)
    fh.close()    

def get_algorithm_result(
    X_train,Y_train,
    X_test,Y_test,
    X_external,Y_external,
    class_names,
    n_splits,n_repeats,
    classifier,clf_name, 
    best_threshold = False,
    dir_result='./'
):
    """
    运行单一算法，给出各种结果:
        内部训练集上:
            P次K折交叉验证的roc曲线
        内部测试集上：
            roc曲线
            混淆矩阵（３个）
        外部数据集上：
            roc曲线
            混淆矩阵（３个）
    """
    dir_result = os.path.join(dir_result,clf_name)
    if not os.path.isdir(dir_result):
        os.mkdir(dir_result)
        
    #　1、获取训练验证集上的roc,auc
    ret = run_RepeatedKFold(n_splits,n_repeats,classifier=classifier,X=X_train,Y=Y_train,
                            title='ROC Curve of '+clf_name+' Classifier \n'+str(n_repeats)+' Times '+str(n_splits)+' Fold Cross Validation',
                            dir_result=dir_result
                           ) 


    classifier.fit(X_train,Y_train)
    if best_threshold:
        dst = np.sqrt(ret['mean_fpr']**2+(ret['mean_tpr']-1)**2)
        best_threshold = ret['mean_threshold'][dst==dst.min()][0]
    
    # 2、校正图
    try:
        fig,calibrated_classifier = plot_calibration_curve_binary_class(
            clf = classifier, 
            clf_name = clf_name, 
            X_train = X_train, 
            y_train = Y_train,
            X_test = X_test, 
            y_test = Y_test,
            cv = 'prefit',
            n_bins = 10,
            dir_result = dir_result
        )
    except :
        calibrated_classifier = classifier
        print("fail to run plot_calibration_curve_binary_class")
        
    # 3、获取测试集上的roc,auc,混淆矩阵,report
    ret_test = get_algorithm_test_result(
        X = X_test,
        Y = Y_test,
        class_names = class_names,
        classifier_fitted = classifier,#calibrated_classifier,#classifier,
        best_threshold = best_threshold,
        clf_name = clf_name,
        dataset_name = 'Test Set',
        dir_result = dir_result
        )
    
    #　结果变量保存
    ret['test'] = ret_test
    
    
    # # 画决策边界
    # visualize_reduced_decision_boundary(
    #     clf=LogR,data=Data,
    #     title="Decision-Boundary of Logistic in dim-Projection",
    #     dir_result=dir_result
    # ) 
    
    ##　外部数据集：
    if X_external and Y_external:
        ret_external = get_algorithm_test_result(
            X = X_external,
            Y = Y_external,
            class_names = class_names,
            classifier_fitted = classifier,
            clf_name = clf_name,
            dataset_name = 'External Data Set',
            dir_result = dir_result
        )
        ret['external'] = ret_external

    return ret,classifier#,calibrated_classifier

from keras import backend as K
import scipy.stats as stats
'''
Compatible with tensorflow backend
'''
def focal_loss(gamma=2., alpha=.25):
    #经过实验，发现loss容易变inf或者NaN
    #参考：https://blog.csdn.net/m0_37477175/article/details/83004746#WCE_Loss_26
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1+0.001))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0+0.001))
    return focal_loss_fixed

def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss
    
    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed


# +
def run_algorithms(data_internal,data_external,
                   feature_select_method='RFE',
                   kbest = 15,
                   n_splits=5,n_repeats=3,
                   best_threshold = False,
                   dir_result='./'):
    """运行多种算法"""
    
    class_names = data_internal['class_names']
#     X = data_internal['X_train'].values
#     Y = data_internal['Y_train'].values.ravel()

    # 特征选择
    Data,feat_imp = select_features(X = data_internal['X_train'],
                                                   Y = data_internal['Y_train'].values,#.ravel(),
                                                   method = feature_select_method,
                                                   dir_result = dir_result,
                                                   kbest = 15)

    name_selected_features = feat_imp.index.values.tolist()

    # 获取数据，切分数据集
    X_train, Y_train = data_internal['X_train'][name_selected_features], data_internal['Y_train']
    X_test, Y_test = data_internal['X_test'][name_selected_features], data_internal['Y_test']
    if data_external:
        X_external = data_external['X'][name_selected_features], 
        Y_external = data_external['Y']#Y_external = onehot2label(Y_external.values)
#         # 删除外部数据集中包含NaN的病例
#         # idx = X_external.index[np.where(np.isnan(X_external))[0]] #[0]表示行号
#         # notnull()
#         tmp = X_external.notnull().all(axis=1)
#         idx = tmp[tmp==True].index.tolist()
#         X_external = X_external.iloc[idx,:]
#         Y_external = Y_external.iloc[idx,:]
    else:
        X_external, Y_external = None, None
        

    ##　数据降维可视化
    visualize_data_reduced_dimension(
        data={'X':pd.concat([X_train,X_test],axis=0), 
              'Y':np.concatenate((Y_train,Y_test),axis=0),
              'class_names':class_names
             },
        n_dim = 3,
        title="Reduced Dimension Projection of Data selected by "+feature_select_method,
        dir_result=dir_result
    ) 
    if X_external and Y_external:
        visualize_data_reduced_dimension(
            data={'X':X_external, 
                  'Y':Y_external, 
                  'class_names':class_names
                 },
            n_dim = 3,
            title="Reduced Dimension Projection of External Data selected by "+feature_select_method,
            dir_result=dir_result
        )
    
    # 转为numpy
    X_train = X_train.values
    Y_train = Y_train.values.ravel()#
    
    
    # Logistic回归
    from sklearn.linear_model import LogisticRegression as LR
    Logistic_clf = LR(penalty='l2',dual=False,tol=0.0001,C=1.0,fit_intercept=True,
       intercept_scaling=1, random_state=None,
       solver='liblinear',max_iter=100,#multi_class='ovr',
       verbose=0,warm_start=False,n_jobs=1)# solver='newton-cg'  'liblinear'  'lbfgs'  'sag' 'saga'
    param_grid ={ 'class_weight' : ['balanced', None] }
    search = GridSearchCV(estimator=Logistic_clf, param_grid=param_grid, scoring='roc_auc')
    search.fit(X_train, Y_train)
    Logistic_clf_best = search.best_estimator_
    ret_Logistic,cLogistic_clf_best = get_algorithm_result(
        X_train,Y_train,X_test,Y_test,
        X_external,Y_external,
        class_names,
        n_splits,n_repeats,
        classifier = Logistic_clf_best,
        clf_name = 'Logistic',
        best_threshold = best_threshold,
        dir_result=dir_result)

    # LDA==========================
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#     from sklearn.covariance import OAS
    LDA_clf = LinearDiscriminantAnalysis()#solver='eigen')#, covariance_estimator=OAS() )
    ret_LDA,cLDA_clf = get_algorithm_result(
        X_train,Y_train,X_test,Y_test,
        X_external,Y_external,
        class_names,
        n_splits,n_repeats,
        classifier = LDA_clf,
        clf_name = 'LDA',
        best_threshold = best_threshold,
        dir_result=dir_result)
    
    # SVM==========================
    from sklearn.svm import SVC
    SVM_clf = SVC(decision_function_shape='ovr',probability=True)
    param_grid ={ 'kernel' : ['rbf', 'sigmoid'] }#'linear','rbf', 'poly', 'sigmoid'
    search = GridSearchCV(estimator=SVM_clf, param_grid=param_grid, scoring='roc_auc')
    search.fit(X_train, Y_train)
    SVM_clf_best = search.best_estimator_
    ret_SVM,SVM_clf_best = get_algorithm_result(
        X_train,Y_train,X_test,Y_test,
        X_external,Y_external,
        class_names,
        n_splits,n_repeats,
        classifier=SVM_clf_best, 
        clf_name='SVM',
        best_threshold = best_threshold,
        dir_result=dir_result
    )
    
    
    ## KNN分类器==========================
    from sklearn.neighbors import KNeighborsClassifier
    KNN_clf = KNeighborsClassifier(n_neighbors=2,metric="minkowski")
    param_grid = {'weights': ['uniform', 'distance'],
                 'p': [1,2]
                 }
    search = GridSearchCV(estimator=KNN_clf, param_grid=param_grid, scoring='roc_auc')
    search.fit(X_train, Y_train)
    KNN_clf_best = search.best_estimator_
    ret_KNN,KNN_clf_best = get_algorithm_result(
        X_train,Y_train,X_test,Y_test,
        X_external,Y_external,
        class_names,
        n_splits,n_repeats,
        classifier=KNN_clf_best, 
        clf_name='KNN',
        best_threshold = best_threshold,
        dir_result=dir_result
    )
    
    
    # GaussianNB型朴素贝叶斯分类器
    from sklearn.naive_bayes import GaussianNB
    GaussianNB_clf = GaussianNB()
    ret_GaussianNB,GaussianNB_clf = get_algorithm_result(
        X_train,Y_train,X_test,Y_test,
        X_external,Y_external,
        class_names,
        n_splits,n_repeats,
        classifier=GaussianNB_clf, 
        clf_name='GaussianNB',
        best_threshold = best_threshold,
        dir_result=dir_result
    )
    
    
    # 决策树==========================
    from sklearn.tree import DecisionTreeClassifier
    Tree_clf = DecisionTreeClassifier()
    param_grid = {'max_depth': [5, 10, 20],
                 'min_samples_leaf': [2,4,8,16] ,
                 'min_samples_split': [2,4,8,16],
                 'class_weight' : ["balanced",None]
                 }
    search = GridSearchCV(estimator=Tree_clf, param_grid=param_grid, scoring='roc_auc')
    search.fit(X_train, Y_train)
    Tree_clf_best = search.best_estimator_
    ret_Tree,Tree_clf_best = get_algorithm_result(
        X_train,Y_train,X_test,Y_test,
        X_external,Y_external,
        class_names,
        n_splits,n_repeats,
        classifier=Tree_clf_best, 
        clf_name='Decision_Tree',
        best_threshold = best_threshold,
        dir_result=dir_result
    )
    
    # ExtraTrees
    from sklearn.ensemble import ExtraTreesClassifier
    ExtraTrees_clf = ExtraTreesClassifier()
    param_grid = {'n_estimators' : [10,50,100,200]}
    search = GridSearchCV(estimator=ExtraTrees_clf, param_grid=param_grid, scoring='roc_auc')
    search.fit(X_train, Y_train)
    ExtraTrees_clf_best = search.best_estimator_
    ret_ExtraTrees,ExtraTrees_clf_best = get_algorithm_result(
        X_train,Y_train,X_test,Y_test,
        X_external,Y_external,
        class_names,
        n_splits,n_repeats,
        classifier=ExtraTrees_clf_best, 
        clf_name='ExtraTrees',
        best_threshold = best_threshold,
        dir_result=dir_result
    )
    
    
    ## 随机森林========================
    from sklearn.ensemble import RandomForestClassifier
    RandomForest_clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                    max_features='auto', max_leaf_nodes=None,
                    min_impurity_decrease=0.0, min_impurity_split=None,
                    min_weight_fraction_leaf=0.0, n_jobs=-1,
                    oob_score=False, random_state=0, verbose=0, warm_start=False)
    from scipy.stats import uniform
    param_grid = dict(max_depth = uniform(loc=5, scale=10),
                      min_samples_leaf = uniform(loc=0, scale=0.1),
                      min_samples_split = uniform(loc=0, scale=0.1),
                      n_estimators = [10,50,100,200],#[10, 20, 35, 50]
                     )
    clf = RandomizedSearchCV(estimator=RandomForest_clf, 
                             param_distributions=param_grid,
                             n_iter=500,n_jobs = -1,
                             scoring='roc_auc')
    search = clf.fit(X_train, Y_train)
    RandomForest_clf_best = search.best_estimator_
    ret_RandomForest,RandomForest_clf_best = get_algorithm_result(
        X_train,Y_train,X_test,Y_test,
        X_external,Y_external,
        class_names,
        n_splits,n_repeats,
        classifier=RandomForest_clf_best, 
        clf_name= 'RandomForest',
        best_threshold = best_threshold,
        dir_result=dir_result
    )
    
    
    # Bagging
    from sklearn.ensemble import BaggingClassifier
    Bagging_clf = BaggingClassifier()
    param_grid = {'base_estimator': [Logistic_clf_best,
                                     LDA_clf,
                                     SVM_clf_best,
                                     #KNN_clf_best,
                                     #GaussianNB_clf,
                                     Tree_clf_best],
                  'n_estimators' : [10,50,100,200]
                 }
    clf = GridSearchCV(estimator=Bagging_clf, param_grid=param_grid, scoring='roc_auc')
    search = clf.fit(X_train, Y_train)
    Bagging_clf_best = search.best_estimator_
    ret_Bagging,Bagging_clf_best = get_algorithm_result(
        X_train,Y_train,X_test,Y_test,
        X_external,Y_external,
        class_names,
        n_splits,n_repeats,
        classifier=Bagging_clf_best, 
        clf_name='Bagging',
        best_threshold = best_threshold,
        dir_result=dir_result
    )
       
    
    ## Adaboost算法
    from sklearn.ensemble import AdaBoostClassifier
    AdaBoost_clf = AdaBoostClassifier(algorithm='SAMME.R') #base_estimator默认是决策树（深度为1），可修改
    param_dict = dict(base_estimator = [Logistic_clf_best,
                                        SVM_clf_best,
                                        #GaussianNB_clf,
                                        Tree_clf_best],
                      n_estimators = [10,50,100,200],
                     )
    search = GridSearchCV(estimator=AdaBoost_clf, param_grid=param_dict, scoring='roc_auc')
    search.fit(X_train, Y_train)
    AdaBoost_clf_best = search.best_estimator_
    ret_AdaBoost,AdaBoost_clf_best = get_algorithm_result(
        X_train,Y_train,X_test,Y_test,
        X_external,Y_external,
        class_names,
        n_splits,n_repeats,
        classifier=AdaBoost_clf_best, 
        clf_name='AdaBoost',
        best_threshold = best_threshold,
        dir_result=dir_result
    )
    
    
    # GradientBoost
    from sklearn.ensemble import GradientBoostingClassifier
    GradientBoost_clf = GradientBoostingClassifier()   
    param_dict = dict(n_estimators = [10,50,100,200],
                      max_depth = uniform(loc=2, scale=10),
                     )
    clf = RandomizedSearchCV(estimator=GradientBoost_clf, 
                             param_distributions=param_dict, 
                             n_iter=500,n_jobs = -1,
                             scoring='roc_auc')
    search = clf.fit(X_train, Y_train)
    GradientBoost_clf_best = search.best_estimator_
    ret_GradientBoost,GradientBoost_clf_best = get_algorithm_result(
        X_train,Y_train,X_test,Y_test,
        X_external,Y_external,
        class_names,
        n_splits,n_repeats,
        classifier=GradientBoost_clf_best, 
        clf_name='GradientBoost',
        best_threshold = best_threshold,
        dir_result=dir_result
    )
    
    
    # MLP
    from sklearn.neural_network import MLPClassifier
    MLP_clf = MLPClassifier(solver='lbfgs',
                            max_iter=5000, 
                            early_stopping=True, 
                            random_state=12345)
    param_grid = {'activation' : ['identity', 'logistic', 'relu'],
                  'learning_rate' : ['constant', 'invscaling', 'adaptive'],
                  'hidden_layer_sizes': [(32,32),(64,64),(128,128)]
                 }
    search = GridSearchCV(estimator=MLP_clf, param_grid=param_grid, scoring='roc_auc')
    search.fit(X_train, Y_train)
    MLP_clf_best = search.best_estimator_
    ret_MLP,MLP_clf_best = get_algorithm_result(
        X_train,Y_train,X_test,Y_test,
        X_external,Y_external,
        class_names,
        n_splits,n_repeats,
        classifier=MLP_clf_best, 
        clf_name='MLP',
        best_threshold = best_threshold,
        dir_result=dir_result
    )

#     # DNN（本段丢弃）
#     import modelsV5 as models
#     batch_size = len(X_train)
#     max_epoch = 3000
#     from keras.wrappers.scikit_learn import KerasClassifier
#     DNN_clf = KerasClassifier(build_fn=models.DNN, batch_size = batch_size, epochs = max_epoch, verbose=0)
#     param_dist = {'input_node': X.shape[1],
#                   'output_node': 1,
#                   'hidden_layers_node': [ [64,64],[128,128],[256,256] ],
#                   'activation': ['relu', 'sigmoid'],
#                   'dropout_rate': [0.2,0.4],
#                   'L1_reg': [0.002,0.01],#stats.uniform(0.001, 0.01),
#                   'L2_reg': [0.002,0.01],#stats.uniform(0.001, 0.01),
#                   'loss_function': ['binary_crossentropy', binary_focal_loss(gamma=2, alpha=0.25)],
#                   'optimizer': ['adam']
#                  }
#     grid_search = GridSearchCV(estimator=DNN_clf, param_grid=param_dist, scoring='roc_auc', n_jobs=1)
#     grid_search.fit(X_train, Y_train)  
#     best_candidate_index = np.flatnonzero(grid_search.cv_results_['rank_test_score'] == 1)[0]
#     best_papram = grid_search.cv_results_['params'][best_candidate_index]
#     DNN_clf_best = grid_search.best_estimator_
#     ret_DNN,best_threshold_DNN = get_algorithm_result(
#         X,Y,X_train,Y_train,X_test,Y_test,
#         n_splits,n_repeats,
#         classifier=DNN_clf_best, 
#         clf_name = 'DNN',
#         dir_result=dir_result)                         



# ##     # DNN
#     Data = {"X_train":X_train,
#            "Y_train":Y_train,
#            "X_test":X_test,
#            "Y_test":Y_test}
#     import modelsV5 as models
#     dir_result_DNN = os.path.join(dir_result,'DNN')
#     dir_result_DNN_train_record = os.path.join(dir_result_DNN,'train_record')   
#     checkpoint_path = os.path.join(dir_result_DNN_train_record,"best_weights.h5")
#     if os.path.isdir(dir_result_DNN_train_record):
#         import shutil
#         shutil.rmtree(dir_result_DNN_train_record)#删除文件夹
#     os.makedirs(dir_result_DNN_train_record)
#     # 训练DNN模型 
#     input_nodes = Data['X_train'].shape[1]
#     output_nodes = 1 if np.unique(Data['Y_train']).size==2 else np.unique(Data['Y_train']).size
#     hidden_layers_node = MLP_clf_best.hidden_layer_sizes
#     max_epoch = 5000
#     DNN = models.dnn(
#         Data,
#         checkpoint_path,
#         input_nodes, 
#         output_nodes, 
#         hidden_layers_node = hidden_layers_node,#[32,32], #[256,256,128]#[256,256] #[64,32,16]
#         max_epoch = max_epoch,
#         activation='sigmoid',#'tanh',#'selu',#'sigmoid',search发现sigmoid>selu 
#         # 实践发现：
#         # 1、学习率过大，则模式坍缩
#         # 2、binary_focal_loss(gamma=2, alpha=0.25)搭配0.001的学习率、hidden_layers_node=[32,32]能得到0.90的AUC
#         # 3、model的fit方法中，class_weights=None或非None，并不影响AUC
#         optimizer='adagrad',#'adadelta',
#         loss_function=[binary_focal_loss(gamma=2, alpha=0.25)],
#         #[tf.keras.losses.MeanAbsoluteError()],
#         #[tf.keras.losses.BinaryCrossentropy(from_logits=False)],
#         #[focal_loss(alpha=.25, gamma=2)],
#         #[binary_focal_loss(gamma=2, alpha=0.25)],
#         L1_reg=1e-2,#1e-3,
#         L2_reg=1e-2,#1e-3,
#         dropout_rate=0.3#search发现0.4>0.2,0.5
#     )
#     n_splits,n_repeats = 5,3
#     ret_DNN, best_threshold_DNN, fig_DNN, ax_DNN = DNN.run_RepeatedKFold(
#         n_splits,n_repeats,
#         title='ROC Curve of DNN\n 5 Times 3 Fold Cross Validation',
#         dir_result=dir_result_DNN 
#     )
    
#     ret_DNN_test = DNN.test(
#         X_train, Y_train,
#         X_test, Y_test,
#         max_epoch=max_epoch,
#         class_names=class_names,
#         best_threshold=True,
#         dir_result=os.path.join(dir_result_DNN,'Test Set')
#                            )
    

    algorithm_names = [
        'Logistic',
        'LDA',
        'SVM',
        'KNN',
        'GaussianNB',
        'Tree',
        'ExtraTrees',
        'RandomForest',
        'Bagging',
        'AdaBoost', 
        'GradientBoost',
        'MLP',
#         'DNN'
    ]

    best_clf_objects = [
        [Logistic_clf_best],
        [LDA_clf],
        [SVM_clf_best],
        [KNN_clf_best],
        [GaussianNB_clf],
        [Tree_clf_best],
        [ExtraTrees_clf_best],
        [RandomForest_clf_best],
        [Bagging_clf_best],
        [AdaBoost_clf_best], 
        [GradientBoost_clf_best],
        [MLP_clf_best],
#         [DNN]
    ]#各分类算法的最佳模型对象
    
    aucs_RepeatedKFold = np.array([
        ret_Logistic['aucs'],
        ret_LDA['aucs'],
        ret_SVM['aucs'],
        ret_KNN['aucs'],
        ret_GaussianNB['aucs'],
        ret_Tree['aucs'],
        ret_ExtraTrees['aucs'],
        ret_RandomForest['aucs'],
        ret_Bagging['aucs'],
        ret_AdaBoost['aucs'], 
        ret_GradientBoost['aucs'],
        ret_MLP['aucs'],
#         ret_DNN['aucs']
    ])
    
    aucs_testset = np.array([
        ret_Logistic['test']['roc_auc'],
        ret_LDA['test']['roc_auc'],
        ret_SVM['test']['roc_auc'],
        ret_KNN['test']['roc_auc'],
        ret_GaussianNB['test']['roc_auc'],
        ret_Tree['test']['roc_auc'],
        ret_ExtraTrees['test']['roc_auc'],
        ret_RandomForest['test']['roc_auc'],
        ret_Bagging['test']['roc_auc'],
        ret_AdaBoost['test']['roc_auc'], 
        ret_GradientBoost['test']['roc_auc'],
        ret_MLP['test']['roc_auc'],
#         ret_DNN_test['roc_auc']
    ])
    
    if data_external:
        aucs_external = np.array([
            ret_Logistic['external']['auc'],
            ret_LDA['external']['auc'],
            ret_SVM['external']['auc'],
            ret_KNN['external']['auc'],
            ret_GaussianNB['external']['auc'],
            ret_Tree['external']['auc'],
            ret_ExtraTrees['external']['auc'],
            ret_RandomForest['external']['auc'],
            ret_Bagging['external']['auc'],
            ret_AdaBoost['external']['auc'], 
            ret_GradientBoost['external']['auc'],
            ret_MLP['external']['auc'],
            #ret_DNN['external']['auc']
        ])

#     best_thresholds = np.array([
#         ret_Logistic['best_threshold'],
#         ret_LDA['best_threshold'],
#         ret_SVM['best_threshold'],
#         ret_KNN['best_threshold'],
#         ret_GaussianNB['best_threshold'],
#         ret_Tree['best_threshold'],
#         ret_ExtraTrees['best_threshold'],
#         ret_RandomForest['best_threshold'],
#         ret_Bagging['best_threshold'],
#         ret_AdaBoost['best_threshold'], 
#         ret_GradientBoost['best_threshold'],
#         ret_MLP['best_threshold'],
# #         best_threshold_DNN#ret_DNN['best_threshold']
#     ])[:,np.newaxis]

#     import pdb
#     pdb.set_trace()
    aucs_RepeatedKFold_df = pd.DataFrame(data=aucs_RepeatedKFold,
                                         index=algorithm_names,
                                         columns=np.arange(aucs_RepeatedKFold.shape[1])
                                        )
    aucs_testset_df = pd.DataFrame(data=aucs_testset,
                                   index=algorithm_names,
                                   columns=[feature_select_method]
                                  )
    if data_external:
        aucs_external_df = pd.DataFrame(data=aucs_external,
                                        index=algorithm_names,
                                        columns=[feature_select_method]
                                       )        
    else:
        aucs_external_df = None
        

    best_clf_objects_df = pd.DataFrame(data=best_clf_objects, columns=['best_clf'], index=algorithm_names)
#     best_thresholds_df = pd.DataFrame(data=best_thresholds, columns=['best_threshold'], index=algorithm_names)
#     best_clfs_df = pd.concat([best_thresholds_df, best_clf_objects_df],axis=1)
    
    fg = boxplot(
        x=[ret_Logistic['aucs'],
           ret_LDA['aucs'],
           ret_SVM['aucs'],
           ret_KNN['aucs'],
           ret_GaussianNB['aucs'],
           ret_Tree['aucs'],
           ret_ExtraTrees['aucs'],
           ret_RandomForest['aucs'],
           ret_Bagging['aucs'],
           ret_AdaBoost['aucs'], 
           ret_GradientBoost['aucs'], 
           ret_MLP['aucs'],
#            ret_DNN['aucs']
          ],
        x_names=algorithm_names,
        title='AUC of Different Algorithms',
        fn_save=os.path.join(dir_result,'AUC-of-Different-Algorithms.png')
    )
    
    return aucs_RepeatedKFold_df, aucs_testset_df, aucs_external_df, best_clf_objects_df, name_selected_features


# -

def run_one_algorithm(data_internal,
                   data_external,
                   method=None,
                   kbest = 15,
                   n_splits=5,n_repeats=3,
                   best_threshold = False,
                   dir_result='./'):
    """
    运行1种算法组合.
    method: tuple, eg.('RFE','SVM')
    """
    
    feature_select_method = method[0]
    ML_method = method[1]
    
    class_names = data_internal['class_names']

    # 特征选择
    Data,feat_imp = select_features(
        X = data_internal['X_train'],
        Y = data_internal['Y_train'].values,#.ravel(),
        method = feature_select_method,
        dir_result = dir_result,
        kbest = kbest)

    name_selected_features = feat_imp.index.values.tolist()

    # 获取数据，切分数据集
    X_train, Y_train = data_internal['X_train'][name_selected_features], data_internal['Y_train']
    X_test, Y_test = data_internal['X_test'][name_selected_features], data_internal['Y_test']
    if data_external:
        X_external = data_external['X'][name_selected_features], 
        Y_external = data_external['Y']#Y_external = onehot2label(Y_external.values)
    else:
        X_external, Y_external = None, None
    
    # 转为numpy
    X_train = X_train.values
    Y_train = Y_train.values.ravel()#
    
    if ML_method in ['Bagging','AdaBoost']:
        # Logistic回归
        from sklearn.linear_model import LogisticRegression as LR
        clf = LR(penalty='l2',dual=False,tol=0.0001,C=1.0,fit_intercept=True,
                 intercept_scaling=1, random_state=None,
                 solver='liblinear',max_iter=100,#multi_class='ovr',
                 verbose=0,warm_start=False,n_jobs=1)# solver='newton-cg'  'liblinear'  'lbfgs'  'sag' 'saga'
        param_grid ={ 'class_weight' : ['balanced', None] }
        search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc')
        search.fit(X_train, Y_train)
        Logistic_clf_best = search.best_estimator_ 

        # LDA==========================
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        LDA_clf = LinearDiscriminantAnalysis()

        # SVM==========================
        from sklearn.svm import SVC
        clf = SVC(decision_function_shape='ovr',probability=True)
        param_grid ={ 'kernel' : ['rbf', 'sigmoid'] }#'linear','rbf', 'poly', 'sigmoid'
        search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc')
        search.fit(X_train, Y_train)
        SVM_clf_best = search.best_estimator_
        
        # 决策树==========================
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        param_grid = {'max_depth': [5, 10, 20],
                      'min_samples_leaf': [2,4,8,16],
                      'min_samples_split': [2,4,8,16],
                     }
        search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc')
        search.fit(X_train, Y_train)
        Tree_clf_best = search.best_estimator_
        
    if ML_method=='LR' or ML_method=='Logistic':
        # Logistic回归
        from sklearn.linear_model import LogisticRegression as LR
        clf = LR(penalty='l2',dual=False,tol=0.0001,C=1.0,fit_intercept=True,
           intercept_scaling=1, random_state=None,
           solver='liblinear',max_iter=100,#multi_class='ovr',
           verbose=0,warm_start=False,n_jobs=1)# solver='newton-cg'  'liblinear'  'lbfgs'  'sag' 'saga'
        param_grid ={ 'class_weight' : ['balanced', None] }
        search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc')
        search.fit(X_train, Y_train)
        clf_best = search.best_estimator_

    
    elif ML_method=='LDA':
        # LDA==========================
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        # from sklearn.covariance import OAS
        clf_best = LinearDiscriminantAnalysis()#solver='eigen')#, covariance_estimator=OAS() )
        
    elif ML_method=='SVM' or ML_method=='SVC':
        # SVM==========================
        from sklearn.svm import SVC
        clf = SVC(decision_function_shape='ovr',probability=True)
        param_grid ={ 'kernel' : ['rbf', 'sigmoid'] }#'linear','rbf', 'poly', 'sigmoid'
        search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc')
        search.fit(X_train, Y_train)
        clf_best = search.best_estimator_
    
    elif ML_method=='KNN':
        ## KNN分类器==========================
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=2,metric="minkowski")
        param_grid = {'weights': ['uniform', 'distance'],
                     'p': [1,2]
                     }
        search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc')
        search.fit(X_train, Y_train)
        clf_best = search.best_estimator_
    
    elif ML_method=='GaussianNB' or ML_method=='GNB':
        # GaussianNB型朴素贝叶斯分类器
        from sklearn.naive_bayes import GaussianNB
        clf_best = GaussianNB()
    
    elif ML_method=='Decision_Tree' or ML_method=='Tree':
        # 决策树==========================
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        param_grid = {'max_depth': [5, 10, 20],
                     'min_samples_leaf': [2,4,8,16] ,
                     'min_samples_split': [2,4,8,16],
                     'class_weight' : ["balanced",None]
                     }
        search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc')
        search.fit(X_train, Y_train)
        clf_best = search.best_estimator_
    
    elif ML_method=='ExtraTrees':
        # ExtraTrees
        from sklearn.ensemble import ExtraTreesClassifier
        clf = ExtraTreesClassifier()
        param_grid = {'n_estimators' : [10,50,100,200]}
        search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc')
        search.fit(X_train, Y_train)
        clf_best = search.best_estimator_
    
    elif ML_method=='RSF' or ML_method=='RandomForestClassifier':
        ## 随机森林========================
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            bootstrap=True, class_weight=None, criterion='gini',
            max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_weight_fraction_leaf=0.0, n_jobs=-1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)
        from scipy.stats import uniform
        param_grid = {'max_depth': uniform(loc=5, scale=10),
                      'min_samples_leaf': uniform(loc=0, scale=0.1),
                      'min_samples_split': uniform(loc=0, scale=0.1),
                      'n_estimators': [10,50,100,200],#[10, 20, 35, 50]
                     }
        clf = RandomizedSearchCV(estimator=clf, 
                                 param_distributions=param_grid,
                                 n_iter=500,
                                 n_jobs = -1,
                                 scoring='roc_auc')
        search = clf.fit(X_train, Y_train)
        clf_best = search.best_estimator_

    elif ML_method=='Bagging':
        ## Bagging========================
        from sklearn.ensemble import BaggingClassifier
        clf = BaggingClassifier()
        param_grid = {'base_estimator': [Logistic_clf_best, LDA_clf, SVM_clf_best, Tree_clf_best],
                      'n_estimators' : [10,50,100,200]
                     }
        clf = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc')
        search = clf.fit(X_train, Y_train)
        clf_best = search.best_estimator_
    
    elif ML_method=='AdaBoost':
        ## Adaboost算法========================
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(algorithm='SAMME.R') #base_estimator默认是决策树（深度为1），可修改
        param_dict = {'base_estimator': [Logistic_clf_best, LDA_clf, SVM_clf_best, Tree_clf_best],
                      'n_estimators': [10,50,100,200],
                     }
        search = GridSearchCV(estimator=clf, param_grid=param_dict, scoring='roc_auc')
        search.fit(X_train, Y_train)
        clf_best = search.best_estimator_
    
    elif ML_method=='GradientBoost': 
        ## GradientBoost========================
        from sklearn.ensemble import GradientBoostingClassifier
        GradientBoost_clf = GradientBoostingClassifier()   
        param_dict = {'n_estimators': [10,50,100,200],
                      'max_depth': uniform(loc=2, scale=10),
                     }
        clf = RandomizedSearchCV(estimator=clf, 
                                 param_distributions=param_dict, 
                                 n_iter=500,n_jobs = -1,
                                 scoring='roc_auc')
        search = clf.fit(X_train, Y_train)
        clf_best = search.best_estimator_
    
    elif ML_method=='MLP':
        ## MLP========================
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(solver='lbfgs',
                                max_iter=5000, 
                                early_stopping=True, 
                                random_state=12345)
        param_grid = {'activation' : ['identity', 'logistic', 'relu'],
                      'learning_rate' : ['constant', 'invscaling', 'adaptive'],
                      'hidden_layer_sizes': [(32,32),(64,64),(128,128)]
                     }
        search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc')
        search.fit(X_train, Y_train)
        clf_best = search.best_estimator_                    

    elif ML_method=='DNN': 
        ## MLP========================
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(solver='lbfgs',
                                max_iter=5000, 
                                early_stopping=True, 
                                random_state=12345)
        param_grid = {'activation' : ['identity', 'logistic', 'relu'],
                      'learning_rate' : ['constant', 'invscaling', 'adaptive'],
                      'hidden_layer_sizes': [(32,32),(64,64),(128,128)]
                     }
        search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc')
        search.fit(X_train, Y_train)
        MLP_clf_best = search.best_estimator_  
        
        ## DNN========================
        Data = {"X_train":X_train,
               "Y_train":Y_train,
               "X_test":X_test,
               "Y_test":Y_test}
        import modelsV5 as models
        dir_result_DNN = os.path.join(dir_result,'DNN')
        dir_result_DNN_train_record = os.path.join(dir_result_DNN,'train_record')   
        checkpoint_path = os.path.join(dir_result_DNN_train_record,"best_weights.h5")
        if os.path.isdir(dir_result_DNN_train_record):
            import shutil
            shutil.rmtree(dir_result_DNN_train_record)#删除文件夹
        os.makedirs(dir_result_DNN_train_record)
        # 训练DNN模型 
        input_nodes = Data['X_train'].shape[1]
        output_nodes = 1 if np.unique(Data['Y_train']).size==2 else np.unique(Data['Y_train']).size
        hidden_layers_node = MLP_clf_best.hidden_layer_sizes
        max_epoch = 5000
        DNN = models.dnn(
            Data,
            checkpoint_path,
            input_nodes, 
            output_nodes, 
            hidden_layers_node = hidden_layers_node,#[32,32], #[256,256,128]#[256,256] #[64,32,16]
            max_epoch = max_epoch,
            activation='sigmoid',#'tanh',#'selu',#'sigmoid',search发现sigmoid>selu 
            # 实践发现：
            # 1、学习率过大，则模式坍缩
            # 2、binary_focal_loss(gamma=2, alpha=0.25)搭配0.001的学习率、hidden_layers_node=[32,32]能得到0.90的AUC
            # 3、model的fit方法中，class_weights=None或非None，并不影响AUC
            optimizer='adagrad',#'adadelta',
            loss_function=[binary_focal_loss(gamma=2, alpha=0.25)],
            #[tf.keras.losses.MeanAbsoluteError()],
            #[tf.keras.losses.BinaryCrossentropy(from_logits=False)],
            #[focal_loss(alpha=.25, gamma=2)],
            #[binary_focal_loss(gamma=2, alpha=0.25)],
            L1_reg=1e-2,#1e-3,
            L2_reg=1e-2,#1e-3,
            dropout_rate=0.3#search发现0.4>0.2,0.5
        )
        ret_DNN, best_threshold_DNN, fig_DNN, ax_DNN = DNN.run_RepeatedKFold(
            n_splits,n_repeats,
            title='ROC Curve of DNN\n 5 Times 3 Fold Cross Validation',
            dir_result=dir_result_DNN 
        )

        ret_DNN_test = DNN.test(
            X_train, Y_train,
            X_test, Y_test,
            max_epoch=max_epoch,
            class_names=class_names,
            best_threshold=True,
            dir_result=os.path.join(dir_result_DNN,'Test Set')
        )
        ret = {'ret_DNN':ret_DNN, 'ret_DNN_test':ret_DNN_test}
        clf_best = DNN
        
    if ML_method!='DNN':
        ret,clf_best = get_algorithm_result(
            X_train,Y_train,X_test,Y_test,
            X_external,Y_external,
            class_names,
            n_splits,n_repeats,
            classifier = clf_best,
            clf_name = ML_method,
            best_threshold = best_threshold,
            dir_result=dir_result)   
        
    metrics = ret
    return metrics, clf_best, name_selected_features
