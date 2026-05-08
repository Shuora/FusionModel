# 融合方式: attention+stacking (two_level_blender)

**Test Accuracy:** 0.9837

**Macro F1:** 0.9580

**分类报告:**

              precision    recall  f1-score   support

           0     0.9750    0.9986    0.9867      2098
           1     0.9211    0.9100    0.9155      1078
           2     0.9888    0.9792    0.9840      3229
           3     0.9320    0.9200    0.9260      1073
           4     0.9972    0.9975    0.9973     10731
           5     0.9650    0.9880    0.9764      3158
           6     0.9988    0.9989    0.9988     10683

    accuracy                         0.9868     32050
   macro avg     0.9683    0.9703    0.9580     32050
weighted avg     0.9875    0.9868    0.9870     32050


**混淆矩阵:**

[[ 2095     0     0     3     0     0     0]
 [   18   981    12    35     5    22     5]
 [    6     4  3162    25     5    25     2]
 [   15    35    12   987     6    12     6]
 [    0     8     3     2 10704    14     0]
 [    4    12     0    10     0  3132     0]
 [    4     2     0     6     0     0 10671]]

![Confusion Matrix](confusion_matrix_two_level_blender.png)
![Metrics Curve](metrics_curve.png)
