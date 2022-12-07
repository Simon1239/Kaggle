# -*- coding: utf-8 -*-
# Author   : ZhangQing
# Time     : 2022/12/7 17:21
# File     : JSP.py
# Project  : 遗传算法解决作业调度问题（JSP）
# Desc     :

import matplotlib.pyplot as plt
from collections import OrderedDict
import random
import copy
import datetime
import numpy as np
import re

jobs = 25  # 工件数
machines = 10  # 机器数
population_num = 10  # 种群规模
population = []  # 初始种群
variation_rate = 0.8  # 变异率
iters = 50  # 进化次数
target_points = [1, 2, 3]  # 变异靶点

# 初始化一个种群
for i in range(population_num):
    population.append(random.sample(range(1, jobs + 1), jobs))


'''time_table = [[31, 41, 25, 30],  # 加工时间表,工件为行，机器为列
              [19, 55, 3, 34],
              [23, 42, 27, 6],
              [13, 22, 14, 13],
              [33, 5, 57, 19]]'''
time_table = np.random.randint(1, 100, (jobs, machines))  # 加工时间表，10*5
# 产生随机解
random.seed(0)


# 定义工作节点类 name为Cij:第i个工件在第j个机器上加工，StartTime为开始时间，LoadTime为加工时间，EndTime为加工结束时间
class Cij:
    def __init__(self, name, StartTime, LoadTime):
        self.name = name
        self.StartTime = StartTime
        self.LoadTime = LoadTime
        self.EndTime = StartTime + LoadTime



# 定义最大流程时间函数
def c_max(n):
    # 循环赋值函数，将工件数，机器数与加工时间进行绑定
    for job, i in enumerate(time_table):
        for machine, loadtime in enumerate(i):
            locals()['c{}_{}'.format(job + 1, machine + 1)] = Cij(name='c{}_{}'.format(job + 1, machine + 1),
                                                                  StartTime=0, LoadTime=loadtime, )
    # 加工流程记录表
    load_time_tables = []
    for num, job in enumerate(n):
        for machine in range(machines + 1):
            if num == 0 and machine == 1:
                locals()['c{}_{}'.format(job, machine)].StartTime = 0
                locals()['c{}_{}'.format(job, machine)].EndTime = locals()['c{}_{}'.format(job, machine)].StartTime + \
                                                                  locals()['c{}_{}'.format(job, machine)].LoadTime
                load_time_tables.append([locals()['c{}_{}'.format(job, machine)].name, [
                    locals()['c{}_{}'.format(job, machine)].StartTime,
                    locals()['c{}_{}'.format(job, machine)].EndTime]])
            elif num == 0 and machine > 1:
                locals()['c{}_{}'.format(job, machine)].StartTime = locals()['c{}_{}'.format(job, machine - 1)].EndTime
                locals()['c{}_{}'.format(job, machine)].EndTime = locals()['c{}_{}'.format(job, machine)].StartTime + \
                                                                  locals()['c{}_{}'.format(job, machine)].LoadTime
                load_time_tables.append([locals()['c{}_{}'.format(job, machine)].name, [
                    locals()['c{}_{}'.format(job, machine)].StartTime,
                    locals()['c{}_{}'.format(job, machine)].EndTime]])
            elif num > 0 and machine == 1:
                locals()['c{}_{}'.format(job, machine)].StartTime = locals()[
                    'c{}_{}'.format(n[num - 1], machine)].EndTime
                locals()['c{}_{}'.format(job, machine)].EndTime = locals()['c{}_{}'.format(job, machine)].StartTime + \
                                                                  locals()['c{}_{}'.format(job, machine)].LoadTime
                load_time_tables.append([locals()['c{}_{}'.format(job, machine)].name, [
                    locals()['c{}_{}'.format(job, machine)].StartTime,
                    locals()['c{}_{}'.format(job, machine)].EndTime]])

            elif num > 0 and machine > 1:
                locals()['c{}_{}'.format(job, machine)].StartTime = max(
                    locals()['c{}_{}'.format(n[num - 1], machine)].EndTime,
                    locals()['c{}_{}'.format(job, machine - 1)].EndTime)
                locals()['c{}_{}'.format(job, machine)].EndTime = locals()['c{}_{}'.format(job, machine)].StartTime + \
                                                                  locals()['c{}_{}'.format(job, machine)].LoadTime
                load_time_tables.append([locals()['c{}_{}'.format(job, machine)].name, [
                    locals()['c{}_{}'.format(job, machine)].StartTime,
                    locals()['c{}_{}'.format(job, machine)].EndTime]])
    return load_time_tables, load_time_tables[-1][-1][-1]


def fitness(n):
    return 1 / (c_max(n)[1])


# 定义节点类state为当前解向量，封装当前解的排列方式，加工时间状况，最大加工时间以及适应度
class node:
    def __init__(self, state):
        self.state = state
        self.load_table = c_max(state)[0]
        self.makespan = c_max(state)[1]
        self.fitness = fitness(state)


# PMX两点交叉函数
def two_points_cross(a1, a2):
    # 不改变原始数据进行操作
    a1_1 = copy.deepcopy(a1)
    a2_1 = copy.deepcopy(a2)
    # 交叉位置,point1<point2
    point1 = random.randint(0, len(a1_1))
    point2 = random.randint(0., len(a1_1))
    while point1 > point2 or point1 == point2:
        point1 = random.randint(0, len(a1_1))
        point2 = random.randint(0., len(a1_1))
    # 记录交叉项
    fragment1 = a1[point1:point2]
    fragment2 = a2[point1:point2]
    # 交叉
    a1_1[point1:point2], a2_1[point1:point2] = a2_1[point1:point2], a1_1[point1:point2]
    # 定义容器
    a1_2 = []  # 储存修正后的head
    a2_2 = []
    a1_3 = []  # 修正后的tail
    a2_3 = []
    # 子代1头部修正
    for i in a1_1[:point1]:
        while i in fragment2:
            i = fragment1[fragment2.index(i)]
        a1_2.append(i)
    # 子代2尾部修正
    for i in a1_1[point2:]:
        while i in fragment2:
            i = fragment1[fragment2.index(i)]
        a1_3.append(i)
    # 子代2头部修订
    for i in a2_1[:point1]:
        while i in fragment1:
            i = fragment2[fragment1.index(i)]
        a2_2.append(i)
    # 子代2尾部修订
    for i in a2_1[point2:]:
        while i in fragment1:
            i = fragment2[fragment1.index(i)]
        a2_3.append(i)

    child1 = a1_2 + fragment2 + a1_3
    child2 = a2_2 + fragment1 + a2_3
    # print('修正后的子代为:\n{}\n{}'.format(child1, child2))
    return child1, child2


######变异函数######

# 交换变异
def gene_exchange(n):
    point1 = random.randint(0, len(n) - 1)
    point2 = random.randint(0, len(n) - 1)
    while point1 == point2 or point1 > point2:
        point1 = random.randint(0, len(n) - 1)
        point2 = random.randint(0, len(n) - 1)
    n[point1], n[point2] = n[point2], n[point1]
    return n


# 插入变异
def gene_insertion(n):
    point1 = random.randint(0, len(n) - 1)
    point2 = random.randint(0, len(n) - 1)
    while point1 == point2:
        point1 = random.randint(0, len(n) - 1)
        point2 = random.randint(0, len(n) - 1)
    x = n.pop(point1)
    n.insert(point2, x)
    return n


# 局部逆序变异
def gene_reverse(n):
    point1 = random.randint(0, len(n) - 1)
    point2 = random.randint(0, len(n) - 1)
    while point1 == point2 or point1 > point2:
        point1 = random.randint(0, len(n) - 1)
        point2 = random.randint(0, len(n) - 1)
    ls_res = n[point1:point2]
    ls_res.reverse()
    l1 = n[:point1]
    l2 = n[point2:]
    n_res_end = l1 + ls_res + l2
    return n_res_end


solution_list = []  # 存放noda解结点类
for i in population:
    locals()['solution{}'.format(population.index(i))] = node(i)
    solution_list.append(locals()['solution{}'.format(population.index(i))])
# 循环开始
start = datetime.datetime.now()
for i in range(iters):
    if i % 10 == 0:
        print('第{}次进化后的最优加工时间为{}'.format(i, solution_list[0].makespan))
    pops = [i.state for i in solution_list]
    pop_children1 = pops[1::2]  # 偶数解
    pop_children2 = pops[::2]  # 奇数解
    # PMX两点交叉变异
    for i in range(len(pop_children1)):
        pop_children1[i], pop_children2[i] = two_points_cross(pop_children1[i], pop_children2[i])
    # 交叉后的子种群
    cross_population = pop_children1 + pop_children2
    # 变异
    for i in cross_population:
        mutation_rate = random.random()
        if mutation_rate > variation_rate:
            target = random.choice(target_points)
            if target == 1:
                cross_population[cross_population.index(i)] = gene_exchange(i)
            elif target == 2:
                cross_population[cross_population.index(i)] = gene_insertion(i)
            else:
                cross_population[cross_population.index(i)] = gene_reverse(i)
    # 选择
    cross_solution = [node(i) for i in cross_population]
    solution_list = solution_list + cross_solution
    solution_list.sort(key=lambda x: x.makespan)
    del solution_list[population_num:]
print('进化完成，最终最优加工时间为：', solution_list[0].makespan)
end = datetime.datetime.now()
print('耗时{}'.format(end - start))
# 绘制甘特图
def color():# 甘特图颜色生成函数
    color_ls=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    col=''
    for i in range(6):
        col+=random.choice(color_ls)
    return '#'+col
colors = [color() for i in range(jobs)]# 甘特图颜色列表
for i in solution_list[0].load_table:
    print(i)
    y=eval(re.findall('_(\d+)',i[0])[0])#正则表达式匹配工件数
    label=re.findall(r'(\d*?)_',i[0])[0]#正则表达式匹配机器数
    plt.barh(y=y, left=i[1][0], width=i[1][-1] - i[1][0], height=0.5, color=colors[eval(label) - 1],
             label=f'job{label}')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.title('jsp最优调度甘特图')
plt.xlabel('加工时间')
plt.ylabel('加工机器')
handles, labels = plt.gca().get_legend_handles_labels()  # 标签去重
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show()