# 融合方式: attention+stacking (two_level_blender)

**Test Accuracy:** 0.9959

**Macro F1:** 0.9924

**分类报告:**

              precision    recall  f1-score   support

           0     0.9992    0.9987    0.9989     16082
           1     0.9828    0.9984    0.9906      3150
           2     0.9848    0.9822    0.9835      3033
           3     1.0000    0.9990    0.9995      3026
           4     0.9831    0.9699    0.9764      3118
           5     0.9987    0.9992    0.9989     15933
           6     0.9981    1.0000    0.9991      1599

    accuracy                         0.9959     45941
   macro avg     0.9924    0.9925    0.9924     45941
weighted avg     0.9959    0.9959    0.9959     45941


**混淆矩阵:**

[[16061     0     0     0     0    21     0]
 [    0  3145     0     0     5     0     0]
 [    0    10  2979     0    44     0     0]
 [    0     0     0  3023     3     0     0]
 [    0    45    46     0  3024     0     3]
 [   13     0     0     0     0 15920     0]
 [    0     0     0     0     0     0  1599]]

![Confusion Matrix](confusion_matrix_two_level_blender.png)
![Metrics Curve](metrics_curve.png)
