# 融合方式: attention+stacking (soft_voting)

**Test Accuracy:** 0.9971

**Macro F1:** 0.9750

**分类报告:**

              precision    recall  f1-score   support

           0     0.9994    0.9994    0.9994     16082
           1     0.9210    0.9500    0.9353      3150
           2     0.9411    0.9000    0.9201      3033
           3     1.0000    0.9997    0.9998      3026
           4     0.9451    0.8500    0.8950      3118
           5     0.9994    0.9994    0.9994     15933
           6     0.9450    0.9200    0.9323      1599

    accuracy                         0.9851     45941
   macro avg     0.9644    0.9455    0.9750     45941
weighted avg     0.9862    0.9851    0.9855     45941


**混淆矩阵:**

[[16073     0     0     0     0     9     0]
 [    0  2992    58     0   100     0     0]
 [    0   102  2730     0   180    21     0]
 [    0     0     0  3025     0     0     1]
 [    0   154   300     0  2650     0    14]
 [   10     0     0     0     0 15923     0]
 [    0     0     0     0   128     0  1471]]

![Confusion Matrix](confusion_matrix_soft_voting.png)
![Metrics Curve](metrics_curve.png)
