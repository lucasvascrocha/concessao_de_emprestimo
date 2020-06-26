# Concessão de empréstimo
(random forest e decision of tree) criação de modelo de machine learning para avaliar a concessão de empréstimo de acordo com características do cliente

**Abstract:**
Dados de uma empresa concessionária de empréstimos que tem como objetivo automatizar a decisão de concessão de empréstimo a seus novos clientes, com isso foi criado um modelo de machine learning que avalia as características dos clientes e prevê se este tem maior probabilidade de pagar ou não seu empréstimo, assim a empresa pode ceder ou não o crédito ao cliente assim como regular a taxa de juros incidente com essa informação.


# Sobre o desenvolvimento

Foram utilizados arquivos disponibilizados pelo Kaggle. O código foi desenvolvido em python e apresentado em jupyter notebook.

# Perguntas a serem respondidas

1. Existe diferença na taxa de juros de um empréstimo para clientes que pagaram seus empréstimos anteriores e clientes que já deixaram de pagá-lo?

-Sim, clientes que nunca atrasaram ou deixaram de pagar seus empréstimos normalmente tem juros menores que os clientes que já atrasaram ou deixaram de pagar seus empréstimos.

2. É possível prever se um cliente irá pagar o empreśtimo ou não de acordo com as características avaliadas?

-Sim, de acordo com algumas caractéristicas como finalidade do empréstimo, parcelas mensais, pontuação de crédito, dentre outras, com uma precisão de 81% de acordo com os dados teinados e testados para este modelo.



# Índice
1 - Load data  
2 - Analysis  
3 - Pre-process  
4 - Modeling  
5 - Evaluate results  

# Requisitos

- Python 3.5
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn
- GridSearchCV

