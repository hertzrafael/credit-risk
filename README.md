895 nulos em person_emp_length

3116 nulos em loan_int_rate

identificação de outliers em person_emp_length

35 registros com idade > percentil 99,9% (66 anos)


## tratamento nulos:
- person_emp_length: peguei a renda anual e coloquei um lower e upper bound de 1500, depois peguei os registros dentro desse limite e calculei a mediana, passando essa mediana para o valor nulo. como fallback, caso não tenha nenhum outro nessa mesma faixa de renda, será aplicada a mediana global do dataset.
- loan_int_rate: Peguei a mediana do grupo (loan_grade)

## tratamento de outliers:
- optei por não remoção, já que é claramente um erro de inserção nos dados
- pessoas com idade maior que 100 anos, terá a idade por idade + person_emp_length (tempo em anos de trabalho)


## resultados:
- primeiro resultado encontrado:
MÉTRICAS DE LightGBM
ROC AUC Calculado: 0.907738979422632
Relatório de Classificação:
              precision    recall  f1-score   support

           0       0.92      0.92      0.92      7657
           1       0.73      0.72      0.72      2118

    accuracy                           0.88      9775
   macro avg       0.82      0.82      0.82      9775
weighted avg       0.88      0.88      0.88      9775

-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
MÉTRICAS DE XGBoost
ROC AUC Calculado: 0.9103300034789522
Relatório de Classificação:
              precision    recall  f1-score   support

           0       0.92      0.93      0.93      7657
           1       0.75      0.71      0.73      2118

    accuracy                           0.89      9775
   macro avg       0.83      0.82      0.83      9775
weighted avg       0.88      0.89      0.88      9775

-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
MÉTRICAS DE RandomForest
ROC AUC Calculado: 0.8985239178898187
Relatório de Classificação:
              precision    recall  f1-score   support

           0       0.92      0.92      0.92      7657
           1       0.73      0.73      0.73      2118

    accuracy                           0.88      9775
   macro avg       0.83      0.83      0.83      9775
weighted avg       0.88      0.88      0.88      9775

-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
MÉTRICAS DE LogisticRegression
ROC AUC Calculado: 0.8425415195881295
Relatório de Classificação:
              precision    recall  f1-score   support

           0       0.91      0.85      0.88      7657
           1       0.56      0.68      0.61      2118

    accuracy                           0.81      9775
   macro avg       0.73      0.77      0.75      9775
weighted avg       0.83      0.81      0.82      9775

-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


