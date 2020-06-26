from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

class Model_select:
    """
    Pré-processamento dos dados
    """

    def __init__(self):
        pass

    def testing_hip_par(self, X_train, y_train):
        """
        Mostra a melhor pontuação AUC de cada modelo após testes de vários hiperparâmetros
        Entrada é o X_train e y_train
        Pode ser lento
        Produzido para problemas de classificação
        """
        # hiperparâmetros para RFC
        n_estimators = [200,300]
        criterion = ['gini', 'entropy']
        param_grid = dict(n_estimators = n_estimators, criterion = criterion )
        rfc = RandomForestClassifier()
        grid = GridSearchCV(estimator=rfc, param_grid = param_grid, scoring='roc_auc',
                            verbose=1, n_jobs=-1, refit=True)
        grid_result = grid.fit(X_train, y_train)
        print('--------------------------------------------------------------------')
        print('Random Forest')
        print('Best Score:' , grid_result.best_score_.round(2))
        print('Best Params:' , grid_result.best_params_)

        # hiperparâmetros para Logistic regression
        C = [1,5,10,50,100]
        penalty = ['l1', 'l2']
        param_grid = dict(C = C, penalty = penalty )
        logreg = LogisticRegression()
        grid = GridSearchCV(estimator=logreg, param_grid = param_grid, scoring='roc_auc',
                            verbose=1, n_jobs=-1, refit = True)
        grid_result = grid.fit(X_train, y_train)
        print('--------------------------------------------------------------------')
        print('Logistic Regression')
        print('Best Score:' , grid_result.best_score_.round(2))
        print('Best Params:' , grid_result.best_params_)

        # hiperparâmetros para Extra tree classifier
        n_estimators = [100,200,300]
        criterion = ["gini", "entropy"]
        max_features = ["auto", "sqrt", "log2"]
        param_grid = dict(n_estimators = n_estimators, criterion = criterion, max_features = max_features )
        etc = ExtraTreesClassifier()
        grid = GridSearchCV(estimator=etc, param_grid = param_grid, scoring='roc_auc',
                            verbose=1, n_jobs=-1, refit = True)
        grid_result = grid.fit(X_train, y_train)
        print('--------------------------------------------------------------------')
        print('Extra tree classifier')
        print('Best Score:' , grid_result.best_score_.round(2))
        print('Best Params:' , grid_result.best_params_)

        # hiperparâmetros para Knn
        n_neighbors = [2,5,10,15,20]
        weights = ['uniform', 'distance']
        algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
        param_grid = dict(n_neighbors = n_neighbors, weights = weights, algorithm = algorithm )
        knn = KNeighborsClassifier()
        grid = GridSearchCV(estimator=knn, param_grid = param_grid, scoring='roc_auc',
                            verbose=1, n_jobs=-1, refit = True)
        grid_result = grid.fit(X_train, y_train)
        print('--------------------------------------------------------------------')
        print('Knn')
        print('Best Score:' , grid_result.best_score_.round(2))
        print('Best Params:' , grid_result.best_params_)

        # hiperparâmetros para Decision of Tree
        criterion = ["gini", "entropy"]
        splitter = ["best", "random"]
        param_grid = dict(criterion = criterion, splitter = splitter)
        decision_tree = DecisionTreeClassifier()
        grid = GridSearchCV(estimator=decision_tree, param_grid = param_grid, scoring='roc_auc',
                            verbose=1, n_jobs=-1, refit = True)
        grid_result = grid.fit(X_train, y_train)
        print('--------------------------------------------------------------------')
        print('Decision of Tree')
        print('Best Score:' , grid_result.best_score_.round(2))
        print('Best Params:' , grid_result.best_params_)

        # hiperparâmetros para bagging_dec_tree
        n_estimators = [2,5,10,20,100,300,500]
        max_samples = [10,50,100,200]
        max_features = [2,3,4,5,6,7,8]
        param_grid = dict(n_estimators = n_estimators, max_samples = max_samples, max_features=max_features)
        bagging_dec_tree = BaggingClassifier(DecisionTreeClassifier(), bootstrap=True)
        grid = GridSearchCV(estimator=bagging_dec_tree, param_grid = param_grid, scoring='roc_auc',
                            verbose=1, n_jobs=-1, refit = True)
        grid_result = grid.fit(X_train, y_train)
        print('--------------------------------------------------------------------')
        print('bagging_dec_tree')
        print('Best Score:' , grid_result.best_score_.round(2))
        print('Best Params:' , grid_result.best_params_)

        # hiperparâmetros para bagging_RFC
        n_estimators = [2,5,10,20,100,300,500]
        max_samples = [10,50,100,200]
        max_features = [2,3,4,5,6,7,8]
        param_grid = dict(n_estimators = n_estimators, max_samples = max_samples, max_features=max_features)
        bagging_RFC = BaggingClassifier(RandomForestClassifier(), bootstrap=True) # roda 300 modelos e cada modelo roda 100 x com   #amostras buscadas aleatoriamente
        grid = GridSearchCV(estimator=bagging_RFC, param_grid = param_grid, scoring='roc_auc',
                            verbose=1, n_jobs=-1, refit = True)
        grid_result = grid.fit(X_train, y_train)
        print('--------------------------------------------------------------------')
        print('bagging_RFC')
        print('Best Score:' , grid_result.best_score_.round(2))
        print('Best Params:' , grid_result.best_params_)

        # hiperparâmetros para SVC
        C = [1,2,5,10,20]
        gamma = ['scale', 'auto']
        param_grid = dict(C = C, gamma=gamma)
        svc_model = SVC(probability=True)
        grid = GridSearchCV(estimator=svc_model, param_grid = param_grid, scoring='roc_auc',
                            verbose=1, n_jobs=-1, refit = True)
        grid_result = grid.fit(X_train, y_train)
        print('--------------------------------------------------------------------')
        print('SVC')
        print('Best Score:' , grid_result.best_score_.round(2))
        print('Best Params:' , grid_result.best_params_)

        # hiperparâmetros para gnb

        var_smoothing = [1e-09,1e-08,1e-07]
        param_grid = dict(var_smoothing = var_smoothing)
        gnb = GaussianNB()
        grid = GridSearchCV(estimator=gnb, param_grid = param_grid, scoring='roc_auc',
                            verbose=1, n_jobs=-1, refit = True)
        grid_result = grid.fit(X_train, y_train)
        print('--------------------------------------------------------------------')
        print('GNB NAIVE BAYES')
        print('Best Score:' , grid_result.best_score_.round(2))
        print('Best Params:' , grid_result.best_params_)



