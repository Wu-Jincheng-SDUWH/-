import numpy as np
###定义单纯形的类
class Simplex(object):
    def __init__(self, c, A, b,mode):
        # 形式 minf(x)=c.Tx
        # s.t. Ax=b
        self.mode = mode  ###True代表求最小值问题，False代表求最大值问题
        self.A = A
        self.b = b
        A = np.array(A)
        b = np.array(b)
        if mode==True:
            c=c
        else:
            c=-1*c
        temp = np.hstack((0, c))
        self.c = c.reshape(-1, 1)
        temp2 = np.hstack((b, A))
        self.mat = np.vstack((temp, np.hstack((b, A))))  ###这里的mat就是初等条件下的矩阵


    def simplex(self):
        c_shape = self.c.shape
        A_shape = self.A.shape
        b_shape = self.b.shape
        end_index = A_shape[1] - A_shape[0]
        N = self.A[:, 0:end_index]
        Not_basic = np.arange(0, end_index)  ###Not_basic是非基变量
        c_Not_basic = self.c[Not_basic, :]
        # 第一个B必须是可逆的矩阵，其实这里应该用算法寻找，但此处省略
        B = self.A[:, end_index:]
        B_columns = np.arange(end_index, A_shape[1])  ###B_columns是基变量
        c_Basic = self.c[B_columns, :]  ###表示去除B_columns中的指定行
        steps = 0
        while True:
            steps += 1
            print("第 {} 步".format(steps))
            optim_achieve, B_columns, Not_basic = self.main_simplex(B, N, c_Basic, c_Not_basic, self.b, B_columns, Not_basic)
            if optim_achieve:
                self.B_columns = B_columns
                self.N_columns = Not_basic
                break
            else:
                B = self.A[:, B_columns]
                N = self.A[:, Not_basic]
                c_Basic = self.c[B_columns, :]
                c_Not_basic = self.c[Not_basic, :]

    def main_simplex(self, B, N, c_B, c_N, b, B_columns, N_columns):
        B_inverse = np.linalg.inv(B)
        P = (c_N.T - np.matmul(np.matmul(c_B.T, B_inverse), N)).flatten()
        Enter_variable = np.argmin(P)
        if P.min() >= 0:
            optim_achieve = True
            print("已经到达极值点")
            best_solution_point = np.matmul(B_inverse, b)
            print("基变量是{}".format(B_columns))
            print("基变量对应的最值点取值是 {}".format(best_solution_point.flatten()))
            if self.mode==True:
                best_value=np.matmul(c_B.T, best_solution_point).flatten()[0]
            else:
                best_value = -1*np.matmul(c_B.T, best_solution_point).flatten()[0]
            print("所取到的最值是 {}".format(best_value))
            return optim_achieve, B_columns, N_columns
        else:
            # 找到进基变量
            Enter_variable = np.argmin(P)
            enter = N[:, Enter_variable].reshape(-1, 1)
            # By=Ni， 求出基
            y = np.matmul(B_inverse, enter)
            pivot_element = np.matmul(B_inverse, b)
            Leave_variable = self.find_pivot_element(y, pivot_element)  ##Leave_variable代表离基变量的序列号
            tmp = N_columns[Enter_variable]
            ##开始进行矩阵运算
            mat = self.mat
            mat = np.array(mat, dtype='float64')
            mat[Leave_variable + 1] = mat[Leave_variable + 1] / mat[Leave_variable + 1][
                Enter_variable + 1]  ##这是因为其实只有两个变量，我补充了4个变量，现在要去掉
            ids = np.arange(mat.shape[0]) != Leave_variable + 1
            mat[ids] -= mat[Leave_variable + 1] * mat[ids,
                                                  Enter_variable + 1:Enter_variable + 2]  # for each i!= row do: mat[i]= mat[i] - mat[row] * mat[i][col]
            print("在这一步中对应的单纯形表为")
            print(mat)
            self.mat = mat
            N_columns[Enter_variable] = B_columns[Leave_variable]
            B_columns[Leave_variable] = tmp
            optim_achieve = False

            print("没有到达极值点")
            print("基变量是{0}，非基变量是{1}".format(sorted(B_columns), sorted(N_columns)))
            return optim_achieve, B_columns, N_columns

    ###找到枢轴元素所在的位置，通过比较x_B/y最小且y>0的位置
    def find_pivot_element(self, y, pivot_element):
        index = []
        min_value = []
        for i, value in enumerate(y):
            if value <= 0:
                continue
            else:
                index.append(i)
                min_value.append(pivot_element[i] / float(value))
        temp=np.argmin(min_value)
        actual_index = index[temp]
        return actual_index
###定义普通单纯形法函数
def Common_simplex(A,b,c,mode):
    simplex = Simplex(c, A, b,mode)
    simplex.simplex()
###定义二阶段法函数
def Two_steps(variable_true,aritifical_variable,A,b,c,obj,mode):
  ##这里的变量个数表示不含人工变量的个数

    simplex = Simplex(c, A, b,True)##这里的mode是固定的，因为第一步都是求最小值问题
    simplex.simplex()
    B_column = simplex.B_columns
    N_column = simplex.N_columns
    B_column = B_column[B_column < variable_true]
    N_column = N_column[N_column < variable_true]
    ###一直到这里一阶段解决了，开始进行二阶段
    ###目标是去掉R1列和R2列和第一行
    obj_row=np.arange(variable_true+1,variable_true+aritifical_variable+1)###这里的obj_row就是R1,和R2所在列，人工变量列都放在了最后
    mat_temp = np.delete(simplex.mat, obj_row, axis=1)
    mat_temp = np.delete(mat_temp, [0], axis=0)
    ##这样得到的mat_temp是去掉人工变量的，下面加上新的目标函数
    ###print(Two_steps.b)
    ###print(Two_steps.mat[0])
    b_temp = mat_temp[:, 0].reshape(-1, 1)
    A_temp = np.delete(mat_temp, 0, axis=1)
    Two_steps = Simplex(obj, A_temp, b_temp,mode)
    B = Two_steps.A[:, B_column]
    N = Two_steps.A[:, N_column]
    c_B = Two_steps.c[B_column, :]
    c_N = Two_steps.c[N_column, :]
    steps = 0
    while True:
        steps += 1
        print("第 {} 步".format(steps))
        is_optim, B_columns, N_columns = Two_steps.main_simplex(B, N, c_B, c_N, Two_steps.b, B_column, N_column)
        if is_optim:
            break
        else:
            B = Two_steps.A[:, B_columns]
            N = Two_steps.A[:, N_columns]
            c_B = Two_steps.c[B_columns, :]
            c_N = Two_steps.c[N_columns, :]
###定义主函数
if __name__ == "__main__":
    ###这里解答Example 3.3
    mode_common=False###题目为解决最大值问题，因此mode为False
    c_common = np.array([5, 4, 0, 0, 0,0])
    A_common = np.array([[6, 4, 1, 0,0, 0], [1, 2, 0, 1,0,0], [-1, 1, 0, 0, 1,0],[0, 1, 0, 0, 0,1]])
    b_common = np.array([24, 6, 1,2]).reshape(-1, 1)
    Common_simplex(A_common,b_common,c_common,mode_common)
    print("Example 3.3 解答完毕")
    ###这里解答Example 3.4
    c = np.array([0, 0, 0, 0, 1, 1])
    A = np.array([[3, 1, 0, 0, 1, 0], [4, 3, -1, 0, 0, 1], [1, 2, 0, 1, 0, 0]])
    b = np.array([3, 6, 4]).reshape(-1, 1)
    obj = np.array([4, 1, 0, 0])
    variable_true=4###这个表示原有变量个数
    aritifical_variable=2##这个表示人工变量的个数
    Two_mode=True
    Two_steps(variable_true,aritifical_variable,A,b,c,obj,Two_mode)
    print("Example 3.4 解答完毕")