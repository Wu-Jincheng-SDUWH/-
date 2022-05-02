from pulp import *

# 1. 建立问题
prob =pulp.LpProblem("Example 3.4.1", LpMinimize)
# 2. 建立变量
x1 = LpVariable("x1", 0)
x2 = LpVariable("x2", 0)

# 3. 设置目标函数
prob += 4*x1 + x2, "Z"
# 4. 施加约束
prob += 3*x1 + x2 == 3
prob += 4*x1 + 3*x2 >= 6
prob +=x1 + 2*x2 <= 4
# 5. 求解
prob.solve()

# 6. 打印求解状态
print("求解状态:", LpStatus[prob.status])

# 7. 打印出每个变量的最优值
for v in prob.variables():
    print(v.name, "=", v.varValue)

# 8. 打印最优解的目标函数值
print("最优解的目标函数值 = ", value(prob.objective))