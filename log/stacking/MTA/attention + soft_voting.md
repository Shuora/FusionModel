# 融合方式: attention+stacking (soft_voting)

**Test Accuracy:** 0.9841

**Macro F1:** 0.9420

**分类报告:**

              precision    recall  f1-score   support

           0     0.9717    0.9990    0.9852      2098
           1     0.8851    0.7801    0.8293      1078
           2     0.9896    0.9681    0.9787      3229
           3     0.8986    0.8005    0.8467      1073
           4     0.9937    0.9991    0.9964     10731
           5     0.9599    0.9921    0.9757      3158
           6     0.9958    0.9991    0.9974     10683

    accuracy                         0.9815     32050
   macro avg     0.9563    0.9340    0.9420     32050
weighted avg     0.9825    0.9815     0.9820     32050


**混淆矩阵:**

[[ 2096     0     0     0     0     0     2]
 [   43   841    32    65    21    63    13]
 [    8     6  3126    24    22    28    15]
 [   45    78    28   859    20    28    15]
 [    0     8     0     0 10721     2     0]
 [    0    18     0     4     3  3133     0]
 [    0     3     3     2     2     0 10673]]

![Confusion Matrix](confusion_matrix_soft_voting.png)
![Metrics Curve](metrics_curve.png)
