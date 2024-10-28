import matplotlib.pyplot as plt

# exercise 1
# depths = [5, 7, 10, 13, 14, 15]
# r2_scores = [
#     0.6476365299851128,  # max_depth=5
#     0.8255940072696436,  # max_depth=7
#     0.9189455906853322,  # max_depth=10
#     0.9285931346983023,  # max_depth=13
#     0.9300320918587416,  # max_depth=14
#     0.9202900074583197   # max_depth=15
# ]

# Exercise 2
# depths = [10, 13, 14]
# r2_scores = [
#     0.7281636586125284,  # max_depth=10
#     0.9509838899744284,  # max_depth=13
#     0.9485945335403348   # max_depth=14
# ]

# # Exercise 4
# depths = [6, 8, 9, 10, 14]
# r2_scores = [
#     0.9261700564720039,  # max_depth=6
#     0.9527531790315265,  # max_depth=8
#     0.9519642618861172,  # max_depth=9
#     0.9479965012094265,  # max_depth=10
#     0.9469323213956408   # max_depth=14
# ]

n_estimators = [30, 50, 80, 100]
r2_scores = [
    0.950808689083078,  # n_estimators=30
    0.9506742574334276,  # n_estimators=50
    0.948503900189452,  # n_estimators=80
    0.9488326261546753, # n_estimators=100
]


# 创建折线图
plt.figure(figsize=(8, 5))
plt.plot(n_estimators, r2_scores, marker='o', linestyle='-', color='b', label='R² Score')

# 添加标签和标题
plt.xlabel('Tree Depth')
plt.ylabel('R² Score')
plt.title('R² Scores at Different Tree Depths (Decision Tree)')
plt.grid(True)
plt.xticks(n_estimators)  # 确保 x 轴刻度与深度对应
plt.legend()

# 显示图表
plt.show()
