import pandas as pd
import numpy as np

class Preprocess:
    """
    Pré-processamento dos dados
    """

    def __init__(self):
        pass

    def find_ouliers(self, df):
        """
        Encontra os outliers
        O DF de entrada precisa ter apenas valores numéricos
        Retorna o print com a % de outl
        """

        # Encontre os outlies, mostre e amazerne os índices
        out_idx = []

        for feature in df.columns:

            # Calule a faxa onde estão os outlies (+- 1.5*IQR)
            Q1 = df[feature].quantile(.25)
            Q3 = df[feature].quantile(.75)
            IQR = Q3 - Q1
            if IQR != 0:

                out_min = Q1 - 1.5 * IQR
                out_max = Q3 + 1.5 * IQR

                # Amarzene os indices dos outlies
                idx = df[feature][(df[feature] <= out_min) | (df[feature] >= out_max)].index.tolist()
                if len(idx) > 0:
                    # Calcule a proporção dos outlies
                    out_p = np.round((len(idx) / len(df)) * 100, 2)
                    out_idx += idx

                # Mostre para inspeção os outlies encontrados
                print("{} ({}%) observações consideradas outlies da variável {}".format(len(idx), out_p, feature))

        # Transfore a lista em um Panda Series
        out_idx = pd.Series(out_idx)

        # Encontre os valores duplicados
        dup_idx = out_idx.iloc[np.where(pd.Series(out_idx).duplicated(keep=False))[0]]
        dup_idx = dup_idx.unique()

        # Encontre os valores únicos
        out_idx = out_idx.unique()

        # Calcule a representação total dos outlies no data set
        out_p = np.round((len(out_idx) / len(df)) * 100, 2)
        dup_p = np.round((len(dup_idx) / len(df)) * 100, 2)

        # Mostre a quantidade total de outlies
        print(
            "No total, {} ({}%) linhas são outliers, sendo {} ({}%) em pelo menos 1 variável(Retirando os duplicados)".format(
                len(out_idx), out_p,
                len(dup_idx), dup_p))

    def convert_to_log(self, df):
        """
        Encontra os outliers e os converte em log
        retorna a lista transformada
        """

        # Encontre os outlies, mostre e amazerne os índices
        out_col_name = []

        for feature in df.columns:

            # Calule a faxa onde estão os outlies (+- 1.5*IQR)
            Q1 = df[feature].quantile(.25)
            Q3 = df[feature].quantile(.75)
            IQR = Q3 - Q1
            if IQR != 0:

                out_min = Q1 - 1.5 * IQR
                out_max = Q3 + 1.5 * IQR

                # Amarzene o nome das features dos outlies
                col_name = df[feature][(df[feature] <= out_min) | (df[feature] >= out_max)].name
                if len(col_name) > 0:
                    out_col_name.append(col_name)

        print(out_col_name)

        for feature in out_col_name:
            df[feature + '_log'] = df[feature].map(lambda i: np.log(i) if i > 0 else 0)

            # transformando negativos em positivos para comparação de skew, o log realmente reduziu o skew?
            if df[feature].skew() < 0:
                a = ((df[feature].skew()) * -1)
            else:
                a = df[feature].skew()

            if df[feature + '_log'].skew() < 0:
                b = ((df[feature + '_log'].skew()) * -1)
            else:
                b = df[feature + '_log'].skew()

            # se a transformação de log aumentar os outliers, dropa a transformação
            if a < b:
                df.drop((f'{feature}_log'), axis=1, inplace=True)

            # se não dropa a original
            else:
                df.drop(feature, axis=1, inplace=True)

        return df

    def cap_by_quantil(self, df):
        """
        Encontra os outliers e os reveste, os menores transformados por quantil(0.10)
        Os maiores por quantil(0.90)
        retorna a lista transformada
        Cuidado * avalie se a  transformação do outlier faz sentido para seus dados
        """

        # Encontre os outlies, mostre e amazerne os índices
        out_col_name = []

        for feature in df.columns:

            # Calule a faxa onde estão os outlies (+- 1.5*IQR)
            Q1 = df[feature].quantile(.25)
            Q3 = df[feature].quantile(.75)
            IQR = Q3 - Q1
            if IQR != 0:

                out_min = Q1 - 1.5 * IQR
                out_max = Q3 + 1.5 * IQR

                # Amarzene o nome das features dos outlies
                col_name = df[feature][(df[feature] <= out_min) | (df[feature] >= out_max)].name
                if len(col_name) > 0:
                    out_col_name.append(col_name)

        print(out_col_name)

        for feature in out_col_name:
            # definindo valores de substituição mínimo e máximo e substituindo-os
            low = df[feature].quantile(0.10)
            high = df[feature].quantile(0.90)

            df[feature] = np.where(df[feature] < low, low, df[feature])
            df[feature] = np.where(df[feature] > high, high, df[feature])

        return df