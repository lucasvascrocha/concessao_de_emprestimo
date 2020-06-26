import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from numpy import set_printoptions
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics


class Feat_select:

    def __init__(self):

        pass

    def rank(self, X, y):
        """
        Entrada de X e y
        Retorna uma lista com combinações de features selecionadas por Random Forest Rank
        10 melhores features em ordem, as 2 melhores, 3 melhores 4 melhores ....
        """

        # estimators
        rf = RandomForestClassifier()
        rf = rf.fit(X, y)
        rfe = RFE(rf, n_features_to_select=1, verbose=2)
        rfe = rfe.fit(X, y)
        rank = pd.DataFrame({'features': X.columns})
        rank['RF rank'] = rfe.ranking_

        rfr = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)
        rfr.fit(X, y)
        rank['RFR'] = (rfr.feature_importances_ * 100)

        linreg = LinearRegression(normalize=True)
        linreg.fit(X, y)
        rank['linreg'] = (linreg.coef_.round(3) * 10)

        model = LogisticRegression(solver='liblinear')
        rfe = RFE(model, 3)
        rfe = rfe.fit(X, y)
        rank['logreg'] = rfe.ranking_

        etc = ExtraTreesClassifier()
        etc.fit(X, y)
        rank['etc'] = (etc.feature_importances_.round(3) * 100)

        test = SelectKBest(score_func=f_classif, k=4)
        fit = test.fit(X, y)
        set_printoptions(precision=3)
        rank['f_score'] = fit.scores_
        print(rank.sort_values('RF rank', ascending=True))

        # opções de listas de features selecionadas para cada estimador
        lista_comb_feat_RFR = []
        lista_comb_feat_RFrank = []
        lista_comb_feat_linreg = []
        lista_comb_feat_logreg = []
        lista_comb_feat_etc = []
        lista_comb_feat_f_score = []
        for x in range(2, len(X.columns)):
            lista_comb_feat_RFR.append(rank.sort_values('RFR', ascending=False).head(x)['features'].tolist())
            lista_comb_feat_RFrank.append(rank.sort_values('RF rank', ascending=True).head(x)['features'].tolist())
            lista_comb_feat_linreg.append(rank.sort_values('linreg', ascending=False).head(x)['features'].tolist())
            lista_comb_feat_logreg.append(rank.sort_values('logreg', ascending=True).head(x)['features'].tolist())
            lista_comb_feat_etc.append(rank.sort_values('etc', ascending=False).head(x)['features'].tolist())
            lista_comb_feat_f_score.append(rank.sort_values('f_score', ascending=False).head(x)['features'].tolist())

        return lista_comb_feat_RFrank

    def test_feat(self, lista_features, df):
        """
        Testa uma lista de features para o modelo kmeans
        Entrada é a lista gerada pela função rank e data frame já préprocessado
        com os dados e todas as features
        Saída é a avaliação AUC para cada combinação de features
        """
        y = df['not.fully.paid'].copy()
        df = df.drop('not.fully.paid', axis=1)
        for x in range(0, len(lista_features)):
            self.df = df[lista_features[x]]

            X_train, X_test, y_train, y_test = train_test_split(self.df, y, test_size=0.30, random_state=101)

            bagging_dec_tree = BaggingClassifier(DecisionTreeClassifier(), max_samples=600, n_estimators=1000,
                                                 bootstrap=True)
            bagging_dec_tree.fit(X_train, y_train)
            predictions = bagging_dec_tree.predict(X_test)

            y_pred_proba = bagging_dec_tree.predict_proba(X_test)[::, 1]
            auc = metrics.roc_auc_score(y_test, y_pred_proba)

            lista_resultados_AUC_por_features = []

            lista_resultados_AUC_por_features.append(
                f'AUC :{round(auc, 2)}- comb feat :{x}')

            print(lista_resultados_AUC_por_features)
            print('--------------------------------------------------------------------------------')

