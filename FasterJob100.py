import os
import glob
import copy
import random
import time
import math
import pandas as pd
from matrix_mult_cython import mathD
from matrix_mult_cython import math_ML
from matrix_mult_cython import initialization0
folder_path = 'C:\\Users\\kmyh\\Desktop\\Benchmark (1)' # 将此路径替换为你的文件夹路径
txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
Txt_Jobs=[]
Txt_t=[]
Txt_Set=[]
Txt_fac=[]
for txt_file in txt_files:
    with open(txt_file, 'r', encoding='utf-8') as file:
        line = file.readline()  # 每次读取一行内容
        if line[0:3] == 'Fac':
            num_fac = int(line[-2])
        a = 0
        m = 0
        Jobs = []
        Process = []
        Set_t = []
        t = 10000000
        job = 10000000
        while line:
            a += 1
            # print(line)  # 默认换行
            line = file.readline()

            if line[0:6] == 'Number':
                t = a + 1
            if line[0:4] == 'Jobs':

                for i in range(0, len(n)):
                    line = file.readline()
                    J = (line.split('\t'))
                    for i in range(len(J) - 1):
                        J[i] = int(J[i])
                    J.pop()
                    Jobs.append(J)
            if a == t:
                n = (line.split('\t'))  # 移除行尾换行符
                for i in range(len(n) - 1):
                    n[i] = int(n[i])
                n.pop()
                for i in range(len(n)):
                    if i > 0:
                        n[i] += n[i - 1]
            if line[0:10] == 'Processing':
                for i in range(0, n[-1]):
                    line = file.readline()
                    P = (line.split('\t'))
                    for i in range(len(P) - 1):
                        P[i] = int(P[i])
                    P.pop()
                    Process.append(P)
            if line[0:10] == 'On machine':
                Set_t0 = []
                for i in range(0, len(n)):
                    line = file.readline()
                    m += 1
                    S2 = (line.split('\t'))
                    for j in range(len(S2) - 1):
                        S2[j] = int(S2[j])
                    S2.pop()
                    Set_t0.append(S2)
                Set_t.append(Set_t0)
        Txt_Jobs.append(Jobs)
        Txt_t.append(Process)
        Txt_Set.append(Set_t)
        Txt_fac.append(num_fac)
def allread(jo,Jobs):
    for i in range(len(Jobs)):
        if jo in Jobs[i]:
            break
    return i

def mathDpr(ti,Pojob):
    D_Pop=[[]for _ in range(len(Pojob))]
    a=0
    for i in Pojob:
        for j in i:
            D=0
            for k in range(1,len(j)):
                D+=mathD(j[k-1],j[k],ti)
            D_Pop[a].append(D)
        a+=1
    return D_Pop

def mathallD(t,Job,old_Job,setup):
    D = []
    C = [[0 for i in range(len(t[0]))] for j in range(len(t))]
    C1 = copy.deepcopy(C)
    time = 0
    Job0=[]
    sit_ML=[0]
    sitML=0

    for i in range(len(Job)):
        sitML+=len(Job[i])
        sit_ML.append(sitML)
        for j in range(len(Job[i])):
            Job0.append(Job[i][j])
    for i in range(0, len(Job0)):
        if i == 0:
            for j in range(0, len(t[0])):
                time += t[Job0[0]][j]
            D.append(time)
        else:
            for k in range(len(t[0]) - 1, -1, -1):
                C[i][k] = t[Job0[i]][k] - t[Job0[i - 1]][k]
                C1[i][k] += t[Job0[i - 1]][k]
                for h in range(len(t[0]) - 1, k - 1, -1):
                    C1[i][k] += C[i][h]
        C1[i].sort(reverse=True)
        if i != 0:
            D.append(C1[i][0])
    for i in range(len(D)):
        if i in sit_ML:
            if i==0:
                D[0] = math_ML(t,setup,Job,old_Job,0,0)[-1]
            else:
                D[i] = math_ML(t, setup, Job, old_Job, sit_ML.index(i)-1, sit_ML.index(i))[-1]

    return D
def mathCmax(t,Jobs,Job_old, Set_t,num_f):
    Cmax_f = []
    for j0 in range(num_f):
        Cmax0 = sum(mathallD(t, Jobs[j0], Job_old, Set_t))
        Cmax_f.append(Cmax0)
    Cmax_num = max(Cmax_f)
    return Cmax_num
def newset(QM,Job):
    for i in range(len(QM[0])):
        QM[0][i] = 1 / Job[-1][-1]
    for i in range(len(Job)):
        numx = Job[i][0]+1
        for j in range(len(Job[i])):
            numy = Job[i][j]+1
            for k in range(len(QM[0])):
                QM[numy][k] = 1 / (len(QM[0]) - len(Job[i]))

        for z in range(len(Job[i])):
            QM[numx][numx + z - 1] = 1 / len(Job[i]) ** 2
            for s in range(len(Job[i])):
                QM[numx + s][numx + z - 1] = 1 / len(Job[i]) ** 2
        QM[i+1][i]=0
    return QM
def find_element_position(lst, element):
    for i, sub_lst in enumerate(lst):
        if element in sub_lst:
            return i
    return None
def PickFirstJob(QM):
    ran=random.uniform(0,1)
    epsilon = 1e-9  # 微小的偏移量
    epsilon0 = 1e-9
    job=0
    if ran==1:
        ran-=epsilon0
    if ran <= epsilon:
        ran += epsilon
    for i in range(0,len(QM[0])):
        if i>0:
            QM[0][i]+=QM[0][i-1]
    for i in range(0,len(QM[0])):
        if i>0:
            if QM[0][i-1]<ran<=QM[0][i]:
                job=i
                break
    return job
def PickJob(QM,lastjob,Jobgroup): #执行完将所选的工件QM置零
    sum_q = 0
    sumq_Matrix = []
    for j in range(Jobgroup[0],Jobgroup[-1]+1):
        sum_q += QM[lastjob +1][j]
        sumq_Matrix.append(sum_q)
    q = random.uniform(0, sum_q)
    for m in range(len(sumq_Matrix)):
        if q <= sumq_Matrix[m]:
            ans=m+Jobgroup[0]
            break
    return ans
def PickGroup(QM,lastjob):
    sum_q = 0
    sumq_Matrix = []
    for j in range(len(QM[0])):
        sum_q += QM[lastjob][j]
        sumq_Matrix.append(sum_q)
    q = random.uniform(0, sum_q)
    for m in range(len(sumq_Matrix)):
        if q <= sumq_Matrix[m]:
            ans = m
            break
    return ans
def UpdateRM(solu,RM):
    for i in range(len(solu)):
        solu[i] = [item for sublist in solu[i] for item in sublist]
        RM[0][solu[i][0]]+=1
        for j in range(len(solu[i])-1):
            RM[solu[i][j]+1][solu[i][j+1]]+=1 #如果工件从0开始 要+1
    return RM
def UpdateQM(QM,RM,r,lenP):
    for i in range(len(QM)):
        for j in range(len(QM[0])):
            QM[i][j]=r*QM[i][j]+(1-r)*RM[i][j]/lenP
    return QM
def delete_QM(QM,lastjob):
    for i in range(len(QM)):
        QM[i][lastjob]=0
    return QM
def SampleQM(QM,Job,fac):
    ans=[]
    for i in range(0,len(Job)):
        zts = []
        if i<fac:
            zts.append(PickFirstJob(QM))
            group=find_element_position(Job,zts[0])
            QM=delete_QM(QM, zts[-1])

            if len(Job[group])>1:
                for j in range(len(Job[group])-1):
                    zts.append( PickJob(QM, zts[-1],Job[group]) )
                    QM=delete_QM(QM,zts[-1])
            ans.append(zts)
        else:
            zts.append(PickGroup(QM,ans[i-1][-1]))
            group = find_element_position(Job, zts[-1])
            QM=delete_QM(QM, zts[-1])
            if len(Job[group])>1:
                for j in range(len(Job[group])-1):
                    zts.append( PickJob(QM, zts[-1],Job[group]) )
                    QM=delete_QM(QM, zts[-1])
            ans.append(zts)
    return ans
def sort_and_get_indices(input_list):
    # 使用sorted()函数对列表元素进行排序，生成一个新的排序后的列表
    sorted_list = sorted(input_list)
    # 使用enumerate()函数同时获取元素和索引，并将它们存储在一个新的列表中
    indices = [index for index, _ in sorted(enumerate(input_list), key=lambda x: x[1])]
    return sorted_list, indices
def sortPop(Pop_fgjob,indices):
    Pop_fgjob0=copy.deepcopy(Pop_fgjob)
    for i in range(len(Pop_fgjob)):
        Pop_fgjob[i]=Pop_fgjob0[indices[i]]
    return Pop_fgjob
def alg_QM(QM,Job,t,set,num_fac,Fjobm,c):

    Pop_job=[]
    Cmax_m = []
    RM = [[0 for _ in range(len(Process))] for _ in range(len(Process) + 1)]
    for _ in range(50):
        QMc=copy.deepcopy(QM)
        Pop_job.append(SampleQM(QMc,Job,num_fac))#没分工厂
    Pop_fgjob=initialization0(Pop_job,t,Job,set,num_fac)
    Pop_fgjob.append(Fjobm)
    for i in range(len(Pop_fgjob)):
        Cmax_m.append(mathCmax(t, Pop_fgjob[i], Job, set, num_fac))
    Cmax_sorted,indices=sort_and_get_indices(Cmax_m)
    Pop_fgjob=sortPop(Pop_fgjob, indices)
    Fgjob=copy.deepcopy(Pop_fgjob)
    HQ_Pop=Fgjob[:len(Fgjob)//3]
    Cmax_b=copy.deepcopy(Cmax_sorted[0])
    Fjob_b=copy.deepcopy(Pop_fgjob[0])

    for i in range(len(HQ_Pop)):
        RM = UpdateRM(HQ_Pop[i], RM)
    QM = UpdateQM(QM, RM, 0.7, len(HQ_Pop))

    return QM,Cmax_b,Fjob_b


def Nsinsert_TSIG1(fgjob,num_fac,Set_t,Jobs,t):
    Cmax_f = []
    for j0 in range(num_fac):
        Cmax0 = sum(mathallD(t, fgjob[j0], Jobs, Set_t))
        Cmax_f.append(Cmax0)
    # 获取每个元素在原始列表中的位置
    Cmax_pos = [(element, index) for index, element in enumerate(Cmax_f)]
    # 对列表进行降序排序
    Cmax_pos.sort(key=lambda x: x[0], reverse=True)
    Cmax_fr = [element for element, _ in Cmax_pos]
    Cmaxc = max(Cmax_f)
    positions = [position for _, position in Cmax_pos]
    new_list = [fgjob[i] for i in positions]
    fgjob = new_list
    target_listc = copy.deepcopy(fgjob)
    s = random.randint(0, len(new_list[0]) - 1)
    group0 = fgjob[0].pop(s)
    for l in range(0, len(fgjob)):
        insert_indices = range(len(fgjob[l]) + 1)
        for k in insert_indices:
            target_list = copy.deepcopy(fgjob)
            target_list[l].insert(k, group0)
            Cmax = mathCmax(t, target_list, Jobs, Set_t, num_fac)
            if Cmaxc >= Cmax:
                Cmaxc = Cmax
                target_listc = target_list

    fgjob = target_listc
    return fgjob,Cmaxc
def Lsinsert_TSIG1(fgjob,num_fac,Set_t,Jobs,t):
    Cmax_f = []
    for j0 in range(num_fac):
        Cmax0 = sum(mathallD(t, fgjob[j0], Jobs, Set_t))
        Cmax_f.append(Cmax0)
    # 获取每个元素在原始列表中的位置
    Cmax_pos = [(element, index) for index, element in enumerate(Cmax_f)]
    # 对列表进行降序排序
    Cmax_pos.sort(key=lambda x: x[0], reverse=True)
    positions = [position for _, position in Cmax_pos]
    new_list = [fgjob[i] for i in positions]
    fgjob = new_list
    target_listc = copy.deepcopy(fgjob)
    Cmaxc = max(Cmax_f)
    s = random.randint(0, len(new_list[0]) - 1)
    if len(new_list[0][s]) > 1:
        Ji = random.randint(0, len(new_list[0][s]) - 1)
        job0 = fgjob[0][s].pop(Ji)
        insert_indices = range(len(fgjob[0][s]) + 1)
        for k in insert_indices:
            target_list = copy.deepcopy(fgjob)
            target_list[0][s].insert(k, job0)
            Cmax = mathCmax(t, target_list, Jobs, Set_t, num_fac)
            if Cmaxc >= Cmax:
                Cmaxc = Cmax
                target_listc = copy.deepcopy(target_list)
        fgjob = target_listc
    return fgjob,Cmaxc
def Nsswap_TSIG2(fgjob,num_fac,Set_t,Jobs,t):
    Cmax_f = []
    for j0 in range(num_fac):
        Cmax0 = sum(mathallD(t, fgjob[j0], Jobs, Set_t))
        Cmax_f.append(Cmax0)
    Cmax_save=max(Cmax_f)
    fgjob_save=copy.deepcopy(fgjob)
    # 获取每个元素在原始列表中的位置
    Cmax_pos = [(element, index) for index, element in enumerate(Cmax_f)]
    # 对列表进行降序排序
    Cmax_pos.sort(key=lambda x: x[0], reverse=True)
    positions = [position for _, position in Cmax_pos]
    fgjob = [fgjob[i] for i in positions]
    s = random.randint(0, len(fgjob[0]) - 1)
    w = random.randint(0, len(fgjob[-1]) - 1)
    group0=fgjob[0].pop(s)
    group1=fgjob[-1].pop(w)
    insert_indices = range(len(fgjob[0]) + 1)
    Cmaxc = float('inf')
    target_listc=copy.deepcopy(fgjob)
    for k in insert_indices:
        target_list = copy.deepcopy(fgjob)
        target_list[0].insert(k, group1)
        Cmax = mathCmax(t, target_list, Jobs, Set_t, num_fac)
        if Cmaxc >= Cmax:
            Cmaxc = Cmax
            target_listc = target_list
    fgjob0=copy.deepcopy(target_listc)
    insert_indices1 = range(len(fgjob[-1]) + 1)
    Cmaxc = float('inf')
    for k in insert_indices1:
        target_list = copy.deepcopy(fgjob0)
        target_list[-1].insert(k, group0)
        Cmax = mathCmax(t, target_list, Jobs, Set_t, num_fac)
        if Cmaxc >= Cmax:
            Cmaxc = Cmax
            target_listc = target_list
    fgjob = target_listc
    if Cmaxc>Cmax_save:
        fgjob=fgjob_save
        Cmaxc=Cmax_save
    return fgjob,Cmaxc
def reinitialization(QMc,QM):
    for i in range(len(QM)):
        for j in range(len(QM[0])):
            QM[i][j]=0.5*QMc[i][j]+0.5*QM[i][j]
    return QM
def Lsfinsert_TSIG1(fgjob,num_fac,Set_t,Jobs,t):
    new_list=copy.deepcopy(fgjob)
    target_listc = copy.deepcopy(fgjob)
    r_fac=random.randint(0,num_fac-1)
    Cmaxc = sum(mathallD(t, new_list[r_fac], Jobs, Set_t))
    Cmax_save = Cmaxc
    fgjob_save = copy.deepcopy(fgjob)
    s = random.randint(0, len(new_list[r_fac]) - 1)
    if len(new_list[r_fac][s]) > 1:
        Ji = random.randint(0, len(new_list[r_fac][s]) - 1)
        job0 = fgjob[r_fac][s].pop(Ji)
        insert_indices = range(len(fgjob[r_fac][s]) + 1)
        for k in insert_indices:
            target_list = copy.deepcopy(fgjob)
            target_list[r_fac][s].insert(k, job0)
            Cmax = sum(mathallD(t, target_list[r_fac], Jobs, Set_t))#改
            if Cmaxc >= Cmax:
                Cmaxc = Cmax
                target_listc = target_list
        fgjob = target_listc
    if Cmaxc>Cmax_save:
        fgjob=fgjob_save
    Cmaxc=mathCmax(t,fgjob,Jobs,Set_t,num_fac)
    return fgjob,Cmaxc
def QMFG(QM, Jobs, Process, Set_t, num_fac, Fjobm,Cmaxm,time_limit):

    CmaxG = [850]
    c=0
    a=0
    A=[0]
    start_t=time.time()
    end_t=time.time()

    while end_t-start_t<time_limit:
        a+=1
        if c<11:
            QM, Cmax_b, Fjob_b= alg_QM(QM, Jobs, Process, Set_t, num_fac, Fjobm,c)  # 把最好的解一直放在种群里
            if Cmaxm > Cmax_b:
                Cmaxm = Cmax_b
                Fjobm = Fjob_b
            for _ in range(2):
                Fjobm, Cmaxm = Nsinsert_TSIG1(Fjobm, num_fac, Set_t, Jobs, Process)
            for _ in range(2):
                Fjobm, Cmaxm = Nsswap_TSIG2(Fjobm, num_fac, Set_t, Jobs, Process)
            for _ in range(2):
                Fjobm, Cmaxm = Lsfinsert_TSIG1(Fjobm, num_fac, Set_t, Jobs, Process)
            for _ in range(2):
                Fjobm, Cmaxm = Lsinsert_TSIG1(Fjobm, num_fac, Set_t, Jobs, Process)

            #for _ in range(2):
                #Fjobm, Cmaxm = LsGJ(Fjobm,num_fac,Set_t,Jobs,Process)
            if Cmaxm > Cmax_b:
                Cmaxm = Cmax_b
                Fjobm = Fjob_b
        elif c>10:
            Cmaxc = copy.deepcopy(Cmaxm)
            Fjobc = copy.deepcopy(Fjobm)
            Fjobmc = copy.deepcopy(Fjobm)
            Fjobm, Cmaxm = TSIG1(Fjobmc, num_fac, Set_t, Jobs, Process)
            for _ in range(4):
                Fjobm, Cmaxm = Lsfinsert_TSIG1(Fjobm, num_fac, Set_t, Jobs, Process)
            for _ in range(4):
                Fjobm, Cmaxm = Lsinsert_TSIG1(Fjobm, num_fac, Set_t, Jobs, Process)
            for _ in range(2):
                Fjobm, Cmaxm = Nsinsert_TSIG1(Fjobm, num_fac, Set_t, Jobs, Process)
            for _ in range(2):
                Fjobm, Cmaxm = Nsswap_TSIG2(Fjobm, num_fac, Set_t, Jobs, Process)
            if Cmaxm>Cmaxc:
                Cmaxm=Cmaxc
                Fjobm=Fjobc

        if Cmaxm >= CmaxG[-1]:
            c += 1
        else:
            c = 0

        A.append(a)
        end_t = time.time()
        CmaxG.append(Cmaxm)


    return Fjobm, Cmaxm
def shuffle_nested_lists(my_list):
    random.shuffle(my_list)
    for sublist in my_list:
        if isinstance(sublist, list):
            shuffle_nested_lists(sublist)
def initialization_Gjob(Process,Jobs,Set_t,num_fac):
    gjob = copy.deepcopy(Jobs)
    gjob = initialization_before0(gjob, Process)
    gjob = initialization_before1(gjob, Process)
    gjob = initialization_before2(gjob, Process)
    fac = [[] for _ in range(num_fac)]
    Cmax_f = [0 for _ in range(num_fac)]
    Cmax_c = copy.deepcopy(Cmax_f)
    for j in range(len(gjob)):
        fac_cop = copy.deepcopy(fac)
        for k in range(len(fac)):
            fac_cop[k].append(gjob[j])
            Cmax_c[k] = sum(mathallD(Process, fac_cop[k], Jobs, Set_t))
        min_diff = float('inf')
        min_index = None
        for i0 in range(len(Cmax_f)):
            diff = abs(Cmax_c[i0] - Cmax_f[i0])
            if diff < min_diff:
                min_diff = diff
                min_index = i0
        fac[min_index].append(gjob[j])  # 对组也要分
    return fac
def initialization_Gjob0(Process,Jobs,Set_t,num_fac):
    gjob = copy.deepcopy(Jobs)
    shuffle_nested_lists(gjob)
    fac = [[] for _ in range(num_fac)]
    Cmax_f = [0 for _ in range(num_fac)]
    Cmax_c = copy.deepcopy(Cmax_f)
    for j in range(len(gjob)):
        fac_cop = copy.deepcopy(fac)
        for k in range(len(fac)):
            fac_cop[k].append(gjob[j])
            Cmax_c[k] = sum(mathallD(Process, fac_cop[k], Jobs, Set_t))
        min_diff = float('inf')
        min_index = None
        for i0 in range(len(Cmax_f)):
            diff = abs(Cmax_c[i0] - Cmax_f[i0])
            if diff < min_diff:
                min_diff = diff
                min_index = i0
        fac[min_index].append(gjob[j])  # 对组也要分
    return fac
def Speedup_insertGroup(facJob,group,Job_old,setup,t):
    if len(facJob)>1:
        facJob_copy = copy.deepcopy(facJob)
        cht = float('inf')
        for i in range(len(facJob) + 1):
            facJob0 = copy.deepcopy(facJob)
            if i == 0:
                facJob0.insert(i, group)
                Cmax0 = mathallD(t, facJob0, Job_old, setup)
                facJob00 = copy.deepcopy(facJob0)
            elif i == len(facJob):
                facJob0.insert(i, group)
                Cmax01 = mathallD(t, facJob0, Job_old, setup)
                if Cmax01 < Cmax0:
                    Cmax0 = Cmax01
                    facJob00 = copy.deepcopy(facJob0)

            else:
                old_t = math_ML(t, setup, facJob0, Job_old, i - 1, i)[-1]
                new_t = Spmath_ML(t, setup, Job_old, group[-1], facJob0[i][0]) + Spmath_ML(t, setup, Job_old,
                                                                                           facJob0[i - 1][-1], group[0])
                chs = new_t - old_t
                if chs < cht:
                    cht = chs
                    i_best = i
        facJob_copy.insert(i_best, group)
        Cmax = mathallD(t, facJob, Job_old, setup)
        if Cmax0 < Cmax:
            Cmax = Cmax0
            facJob_copy = facJob00
    else:
        for i in range(len(facJob) + 1):
            facJob0 = copy.deepcopy(facJob)
            if i == 0:
                facJob0.insert(i, group)
                Cmax0 = mathallD(t, facJob0, Job_old, setup)
                facJob_copy = copy.deepcopy(facJob0)
            elif i == len(facJob):
                facJob0.insert(i, group)
                Cmax01 = mathallD(t, facJob0, Job_old, setup)
                if Cmax01 < Cmax0:
                    Cmax0 = Cmax01
                    facJob_copy = facJob0
    return facJob_copy
def Spmath_ML(time,sd,Job,ed_i,st_i):#ed代表前者组最后一个工件，str代表后者第一个工件
    m1 = len(time[0])
    ML = [0 for _ in range(m1)]
    k = allread(st_i,Job)#后者的组
    k0 = allread(ed_i, Job)
    for j in range(0, m1):
        if j == 1:
            ML[j] = max(sd[0][k0][k] + time[st_i][j - 1] - time[ed_i][j], sd[1][k0][k]) + time[st_i][j]
        elif j > 1:
            ML[j] = max(ML[j - 1] - time[ed_i][j], sd[j][k0][k]) + time[st_i][j]
    return ML[-1]
def TSIG1(fgjob,num_fac,Set_t,Jobs,t):#会把fgjob搞烂
    d=random.randint(2, 7)
    group0=[]
    Cmax_f = []
    for j0 in range(num_fac):
        Cmax0 = sum(mathallD(t, fgjob[j0], Jobs, Set_t))
        Cmax_f.append(Cmax0)
    # 获取每个元素在原始列表中的位置
    Cmax_pos = [(element, index) for index, element in enumerate(Cmax_f)]
    # 对列表进行降序排序
    Cmax_pos.sort(key=lambda x: x[0], reverse=True)
    positions = [position for _, position in Cmax_pos]
    fgjob = [fgjob[i] for i in positions]
    target_listc=copy.deepcopy(fgjob)
    for i in range(0, d):
        if i < num_fac:
            if len(fgjob[i])>1:
                s = random.randint(0, len(fgjob[i]) - 1)
                group0.append(fgjob[i].pop(s))
        else:
            i0=random.randint(0,num_fac-1)
            s = random.randint(0, len(fgjob[i0]) - 1)
            group0.append(fgjob[i0].pop(s))

    for j in group0:
        Cmaxc = float('inf')
        for l in range(0,len(fgjob)):
            target_list = copy.deepcopy(fgjob)
            target_list[l]=Speedup_insertGroup(target_list[l], j, Jobs, Set_t, t)
            Cmaxm=mathCmax(t,target_list,Jobs,Set_t,num_fac)
            if Cmaxm<Cmaxc:
                Cmaxc=Cmaxm
                target_listc=target_list
        fgjob=target_listc

    return fgjob,Cmaxc
def CIG1(fgjob,num_fac,Set_t,Jobs,t):
    d=random.randint(2, 7)
    group0=[]
    Cmax_f = []
    for j0 in range(num_fac):
        Cmax0 = sum(mathallD(t, fgjob[j0], Jobs, Set_t))
        Cmax_f.append(Cmax0)
    # 获取每个元素在原始列表中的位置
    Cmax_pos = [(element, index) for index, element in enumerate(Cmax_f)]
    # 对列表进行降序排序
    Cmax_pos.sort(key=lambda x: x[0], reverse=True)
    positions = [position for _, position in Cmax_pos]
    fgjob = [fgjob[i] for i in positions]
    target_listc=copy.deepcopy(fgjob)
    tis=0
    while tis<d:
        i=random.randint(0,num_fac-1)
        s = random.randint(0, len(fgjob[i]) - 1)
        if len(fgjob[i][s])>1:
            group0.append(fgjob[i].pop(s))
            tis+=1


    for j in group0:
        Cmaxc = float('inf')
        for l in range(0,len(fgjob)):
            for k in range(len(fgjob[l])):
                target_list = copy.deepcopy(fgjob)
                target_list[l].insert(k,j)
                Cmaxm = mathCmax(t, target_list, Jobs, Set_t, num_fac)
                if Cmaxm < Cmaxc:
                    Cmaxc = Cmaxm
                    target_listc = target_list

        fgjob=target_listc

    return fgjob,Cmaxc
def TSIG2(fgjob,num_fac,Set_t,Jobs,t):#会把fgjob搞烂
    d=random.randint(2, 7)
    group0=[]
    Cmax_f = []
    for j0 in range(num_fac):
        Cmax0 = sum(mathallD(t, fgjob[j0], Jobs, Set_t))
        Cmax_f.append(Cmax0)
    # 获取每个元素在原始列表中的位置
    Cmax_pos = [(element, index) for index, element in enumerate(Cmax_f)]
    # 对列表进行降序排序
    Cmax_pos.sort(key=lambda x: x[0], reverse=True)
    positions = [position for _, position in Cmax_pos]
    fgjob = [fgjob[i] for i in positions]
    target_listc = copy.deepcopy(fgjob)
    if d>len(fgjob[0]):
        d=len(fgjob[0])
    for _ in range(d):
        selected_element = random.choice(fgjob[0])
        group0.append(selected_element)
        fgjob[0].remove(selected_element)
    insert_indices = range(len(fgjob[0]) + 1)
    for j in group0:
        Cmaxc=float('inf')
        for k in insert_indices:
            target_list = copy.deepcopy(fgjob)
            target_list[0].insert(k, j)
            Cmax=mathCmax(t,target_list,Jobs,Set_t,num_fac)
            if Cmaxc>=Cmax:
                Cmaxc=Cmax
                target_listc=target_list
        fgjob=target_listc
    return fgjob,Cmaxc
def initialization_before0(Jobs, Process):
    result = [sum(sublist) for sublist in Process]
    for i in range(len(Jobs)):
        if len(Jobs[i]) > 1:
            sorted_jobs = sorted(Jobs[i], key=lambda x: result[Jobs[i].index(x)], reverse=True)
            Jobs[i] = sorted_jobs
    return Jobs
def initialization_before1(Jobs, Process):
    PoG = []
    for i in range(len(Jobs)):
        D0 = sum(Process[Jobs[i][0]])
        if len(Jobs[i]) > 1:
            for j in range(1, len(Jobs[i])):
                D0 += mathD(Jobs[i][j - 1], Jobs[i][j], Process)
        PoG.append(D0)
    Jobs = sorted(Jobs, key=lambda x: PoG[Jobs.index(x)], reverse=True)
    return Jobs
def mathsD(Jobs, Process):
    D0 = sum(Process[Jobs[0]])
    if len(Jobs) > 1:
        for j in range(1, len(Jobs)):
            D0 += mathD(Jobs[j - 1], Jobs[j], Process)
    return D0
def initialization_before2(Jobs,Process):
    Jobsc=copy.deepcopy(Jobs)
    for i in range(len(Jobs)):
        if len(Jobsc[i]) > 1:
            random_list = random.sample(Jobsc[i], 2)
            for element in random_list:
                Jobsc[i].remove(element)
            D0 = sum(Process[random_list[0]])
            D1 = sum(Process[random_list[1]])
            D0 += mathD(random_list[0], random_list[1], Process)
            D1 += mathD(random_list[1], random_list[0], Process)
            if D0 > D1:
                random_list[0], random_list[1] = random_list[1], random_list[0]
            if len(Jobsc[i])>0:
                for j in Jobsc[i]:
                    Cmaxm=float('inf')
                    for l in range(len(random_list)+1):
                        random_listc=copy.deepcopy(random_list)
                        random_listc.insert(l,j)
                        Cmax=mathsD(random_listc,Process)
                        if Cmaxm>Cmax:
                            Cmaxm=Cmax
                            random_list=random_listc
            Jobsc[i]=random_list
    return Jobsc
def alg_mCCEA(Jobs, Process, Set_t, num_fac,time_limit):  # 缺一个重置
    Pop_gjob = []
    Pop_job = []
    Pop_group = []
    Jobs0 = copy.deepcopy(Jobs)
    Jobs0 = initialization_before0(Jobs0, Process)
    Jobs0 = initialization_before1(Jobs0, Process)
    Pop_gjob.append(Jobs0)
    for _ in range(50):  # 种群大小(重置部分还有一个)
        Jobs1 = copy.deepcopy(Jobs)
        shuffle_nested_lists(Jobs1)
        Pop_gjob.append(Jobs1)
    Pop_fgjob = initialization0(Pop_gjob, Process, Jobs, Set_t, num_fac)
    Cmax_p = []
    for i in range(len(Pop_fgjob)):
        Cmax0 = mathCmax(Process, Pop_fgjob[i], Jobs, Set_t, num_fac)
        Cmax_p.append(Cmax0)
    Pop_fgjob = sorted(Pop_fgjob, key=lambda x: Cmax_p[Pop_fgjob.index(x)])  # 从fgjob中提取出Pop_job,Pop_group
    sorted_Cmax = sorted(Cmax_p)
    Cmax_r = sorted_Cmax[0:3]
    sorted_Cmax[0:3] = []
    for zz in range(len(Pop_fgjob)):
        GJobs, Fgroup = GroupJob_separation(Pop_fgjob[zz], Jobs)
        Pop_job.append(GJobs)
        Pop_group.append(Fgroup)
    fgjob_refer = Pop_fgjob[0:3]
    Pop_fgjob[0:3] = []
    job_refer = Pop_job[0:3]
    Pop_job[0:3] = []
    group_refer = Pop_group[0:3]
    Pop_group[0:3] = []
    c = 0

    #time_limit=200#len(Process[0])*len(Jobs)*5
    start_time_i = time.time()
    end_time_i = time.time()

    while end_time_i - start_time_i < time_limit:
        copy_Cmax_r = copy.deepcopy(Cmax_r)
        if c <= 5:
            for ii in range(len(Pop_group)):
                ran_times = random.randint(3, 8)
                for _ in range(ran_times):
                    index_fac0 = random.randint(0, len(Pop_group[ii]) - 1)
                    index_fac1 = random.randint(0, len(Pop_group[ii]) - 1)
                    index_group0 = random.randint(0, len(Pop_group[ii][index_fac0]) - 1)
                    index_group1 = random.randint(0, len(Pop_group[ii][index_fac1]) - 1)
                    Pop_group[ii][index_fac0][index_group0], Pop_group[ii][index_fac1][index_group1] = \
                        Pop_group[ii][index_fac1][index_group1], Pop_group[ii][index_fac0][index_group0]
            for jj in range(len(Pop_job)):
                ran_times = random.randint(3, 8)
                for _ in range(ran_times):
                    index_gjob = random.randint(0, len(Pop_job[jj]) - 1)
                    index_job0 = random.randint(0, len(Pop_job[jj][index_gjob]) - 1)
                    index_job1 = random.randint(0, len(Pop_job[jj][index_gjob]) - 1)
                    Pop_job[jj][index_gjob][index_job0], Pop_job[jj][index_gjob][index_job1] = Pop_job[jj][index_gjob][
                                                                                                   index_job1], \
                                                                                               Pop_job[jj][index_gjob][
                                                                                                   index_job0]
            for k in range(len(Pop_job)):
                gr = random.randint(0, 2)
                fgjob = GroupJob_combination(Pop_job[k], group_refer[gr])  # 局部搜索
                fgjob, Cmaxc = Nsinsert_TSIG1(fgjob, num_fac, Set_t, Jobs, Process)
                fgjob, Cmaxc = Nsswap_TSIG2(fgjob, num_fac, Set_t, Jobs, Process)
                if Cmaxc < sorted_Cmax[k]:
                    sorted_Cmax[k] = Cmaxc
                    Pop_job[k], Pop_group[k] = GroupJob_separation(fgjob, Jobs)
                    if Cmaxc < Cmax_r[gr]:
                        fgjob_refer[gr] = fgjob
                        Cmax_r[gr] = Cmaxc
                        job_refer[gr], group_refer[gr] = GroupJob_separation(fgjob, Jobs)

            if Cmax_r == copy_Cmax_r:
                c += 1
            for k in range(len(Pop_job)):  # 工件作为参考合作
                gr = random.randint(0, 2)
                fgjob = GroupJob_combination(job_refer[gr], Pop_group[k])  # 局部搜索
                fgjob, Cmaxc = Nsinsert_TSIG1(fgjob, num_fac, Set_t, Jobs, Process)
                fgjob, Cmaxc = Nsswap_TSIG2(fgjob, num_fac, Set_t, Jobs, Process)
                if Cmaxc < sorted_Cmax[k]:
                    sorted_Cmax[k] = Cmaxc
                    Pop_job[k], Pop_group[k] = GroupJob_separation(fgjob, Jobs)
                    if Cmaxc < Cmax_r[gr]:
                        fgjob_refer[gr] = fgjob
                        Cmax_r[gr] = Cmaxc
                        job_refer[gr], group_refer[gr] = GroupJob_separation(fgjob, Jobs)
            if Cmax_r == copy_Cmax_r:
                c += 1
        else:
            Pop_gjob = []
            Pop_job = []
            Pop_group = []
            for _ in range(50):
                Jobs1 = copy.deepcopy(Jobs)
                shuffle_nested_lists(Jobs1)
                Pop_gjob.append(Jobs1)
            Pop_fgjob = initialization0(Pop_gjob, Process, Jobs, Set_t, num_fac)
            Cmax_p = []
            for i in range(len(Pop_fgjob)):
                Cmax0 = mathCmax(Process, Pop_fgjob[i], Jobs, Set_t, num_fac)
                Cmax_p.append(Cmax0)
            Pop_fgjob = sorted(Pop_fgjob, key=lambda x: Cmax_p[Pop_fgjob.index(x)])  # 从fgjob中提取出Pop_job,Pop_group
            sorted_Cmax = sorted(Cmax_p)

            for zz in range(len(Pop_fgjob)):
                GJobs, Fgroup = GroupJob_separation(Pop_fgjob[zz], Jobs)
                Pop_job.append(GJobs)
                Pop_group.append(Fgroup)
            c = 0
        end_time_i = time.time()

    return fgjob_refer, Cmax_r
def GroupJob_separation(fgjob, Jobs):
    GJobs = copy.deepcopy(Jobs)
    Fgroup = [[] for _ in range(len(fgjob))]
    for i in range(len(fgjob)):
        for j in range(len(fgjob[i])):
            target_value = fgjob[i][j][0]
            row_index = -1
            for z, row in enumerate(Jobs):
                if target_value in row:
                    row_index = z
                    break
            GJobs[row_index] = fgjob[i][j]
            Fgroup[i].append(row_index)
    return GJobs, Fgroup
def GroupJob_combination(Gjobs, Fgroup):
    fgjob = [[] for _ in range(len(Fgroup))]
    for i in range(len(Fgroup)):
        for j in range(len(Fgroup[i])):
            fgjob[i].append(Gjobs[Fgroup[i][j]])
    return fgjob
def alg_tig(Job,t,set,num_fac,Fjobm):
    Pop_gjob = []
    for _ in range(50):  # 种群大小
        Jobs1 = copy.deepcopy(Jobs)
        shuffle_nested_lists(Jobs1)
        Pop_gjob.append(Jobs1)
    Pop_fgjob = initialization0(Pop_gjob, t, Jobs, Set_t, num_fac)
    Pop_fgjob.append(Fjobm)

    Cmax_m=mathCmax(t,Fjobm,Job,set,num_fac)
    Pop_fgjob0=copy.deepcopy(Pop_fgjob)
    for i in range(len(Pop_fgjob)):
        Fjob_b, Cmax_b = TSIG1(Pop_fgjob0[i],num_fac,set,Job,t)
        Fjob_b, Cmax_b = TSIG2(Fjob_b, num_fac, set, Job, t)
        if Cmax_b<=Cmax_m:
            Fjobm=Fjob_b
            Cmax_m=Cmax_b
    return Fjobm,Cmax_m
def iterated(Jobs, Process, Set_t, num_fac, Fjobm,Cmaxm,time_limit):
    CmaxG = [0]
    c = 0
    #time_limit=len(Process[0])*len(Jobs)*5
    start_time_i=time.time()
    end_time_i=time.time()

    while end_time_i-start_time_i<time_limit:
        Fjob_b, Cmax_b = alg_tig(Jobs, Process, Set_t, num_fac, Fjobm)  # 把最好的解一直放在种群里
        if Cmaxm > Cmax_b:
            Cmaxm = Cmax_b
            Fjobm = Fjob_b
        for _ in range(2):
            Fjobm, Cmaxm = Nsinsert_TSIG1(Fjobm, num_fac, Set_t, Jobs, Process)
        for _ in range(2):
            Fjobm, Cmaxm = Nsswap_TSIG2(Fjobm, num_fac, Set_t, Jobs, Process)
        for _ in range(2):
            Fjobm, Cmaxm = Lsinsert_TSIG1(Fjobm, num_fac, Set_t, Jobs, Process)

        if Cmaxm == CmaxG[-1]:
            c += 1
        end_time_i=time.time()
        CmaxG.append(Cmaxm)
    return Fjobm, Cmaxm
def random_split_number(num, n):
    if n <= 0:
        raise ValueError("n必须大于0")
    if num < n:
        raise ValueError("数字必须大于或等于n")

    # 初始化每份的大小为1
    parts = [1] * n

    # 计算剩余的点数
    remaining_points = num - n

    # 随机分配剩余的点数
    for i in range(remaining_points):
        # 随机选择一份，并增加其大小
        random_part = random.randint(0, n - 1)
        parts[random_part] += 1

    return parts
def CIG2(fgjob,num_fac,Set_t,Jobs,t):
    EF=[]
    EJ=[[]for _ in range(num_fac)]
    Fjobc=copy.deepcopy(fgjob)
    parts = random_split_number(len(Jobs)//2, num_fac)
    for i in range(num_fac):
        num_list=list(range(len(fgjob[i])))
        if len(fgjob[i])<=parts[i]:
            parts[i]=len(fgjob[i])-1
        random_sample = random.sample(num_list, parts[i])
        EF.append(random_sample)
    EFc=copy.deepcopy(EF)
    for i in range(len(EFc)):
        for j in EFc[i]:
            if len(fgjob[i][j])>1:
                random_sample_indices = random.sample(range(len(Fjobc[i][j])), len(Fjobc[i][j])//2)
                # 创建一个新列表，存储抽取的元素
                random_sample = [Fjobc[i][j][f] for f in random_sample_indices]
                # 从原始列表中删除抽取的元素
                Fjobc[i][j] = [element for index, element in enumerate(Fjobc[i][j]) if index not in random_sample_indices]
                EJ[i].append(random_sample)
            else:
                EF[i].pop(EF[i].index(j))
    for i in range(len(EF)):
        rec=0
        if len(EF[i])!=0:
            for j in EF[i]:
                for k in range(len(EJ[i][rec])):
                    Cmaxm = float('inf')
                    for l in range(len(Fjobc[i][j]) + 1):
                        target_list = copy.deepcopy(Fjobc)
                        target_list[i][j].insert(l, EJ[i][rec][k])
                        Cmax = mathCmax(t, target_list, Jobs, Set_t, num_fac)
                        if Cmax <= Cmaxm:
                            target_listc = copy.deepcopy(target_list)
                            Cmaxm = Cmax
                    Fjobc = copy.deepcopy(target_listc)
                rec += 1

    return Fjobc,Cmaxm
def CoIG(Jobs, t, Set_t, num_fac, Fjobm,time_limit):
    Joblter = 3
    Grouplter = 5
    Cmax_m = float('inf')
    endtime=time.time()
    starttime=time.time()
    #time_limit=len(Process[0])*len(Jobs)*5
    while endtime-starttime<time_limit:
        count = 0
        while count < Grouplter and endtime-starttime<time_limit:
            Cmaxm0 = copy.deepcopy(Cmax_m)
            Fjobmc = copy.deepcopy(Fjobm)
            Fjob_b, Cmax_b = CIG1(Fjobmc, num_fac, Set_t, Jobs, t)
            for _ in range(2):
                Fjob_b, Cmax_b = Nsinsert_TSIG1(Fjob_b, num_fac, Set_t, Jobs, t)
            if Cmax_b == Cmaxm0:
                count += 1
            if Cmax_m >= Cmax_b:
                Cmax_m = Cmax_b
                Fjobm = Fjob_b
            endtime = time.time()
        count = 0
        while count < Joblter and endtime-starttime<time_limit:
            Cmaxm0 = copy.deepcopy(Cmax_m)
            Fjobmc = copy.deepcopy(Fjobm)
            Fjob_b, Cmax_b = CIG2(Fjobmc, num_fac, Set_t, Jobs, t)
            if Cmax_b == Cmaxm0:
                count += 1
            if Cmax_m >= Cmax_b:
                Cmax_m = Cmax_b
                Fjobm = Fjob_b
            endtime = time.time()
        endtime=time.time()
    return Fjobm, Cmax_m
def DIWO1(Process,Jobs,Set_t,num_fac,Fjobm,limit_time):
    Pop_gjob = []
    for _ in range(9):
        Jobs1 = copy.deepcopy(Jobs)
        shuffle_nested_lists(Jobs1)
        Pop_gjob.append(Jobs1)
    Pop_fgjob = initialization0(Pop_gjob, Process, Jobs, Set_t, num_fac)

    time_start = time.time()
    end_time = time.time()
    #limit_time=len(Process[0])*len(Jobs)*5
    while end_time-time_start<limit_time*0.9:
        Cmax_m = []
        num_seedlist = []
        Seed=[]
        Pop_fgjob.append(Fjobm)
        for i in range(len(Pop_fgjob)):
            Cmax_m.append(mathCmax(Process, Pop_fgjob[i], Jobs, Set_t, num_fac))
        Cmax_sorted, indices = sort_and_get_indices(Cmax_m)
        Pop_fgjob = sortPop(Pop_fgjob, indices)
        for i in Cmax_sorted:
            if Cmax_sorted[-1] - Cmax_sorted[0] == 0:
                num_seed = 1
            else:
                num_seed = int(1 + ((Cmax_sorted[-1] - i) / (Cmax_sorted[-1] - Cmax_sorted[0])) * 9)  # 参数可更改
            num_seedlist.append(num_seed)
        for i in range(len(Pop_fgjob)):
            for j in range(num_seedlist[i]):
                fgjob = copy.deepcopy(Pop_fgjob[i])
                fgjob0 = copy.deepcopy(fgjob)
                r_f1=random.randint(0,num_fac-1)
                r_g1=random.randint(0,len(fgjob0[r_f1])-1)
                Rj=fgjob0[r_f1].pop(r_g1)
                r_f2 = random.randint(0, num_fac - 1)
                r_g2 = random.randint(0, len(fgjob0[r_f2]) - 1)
                fgjob0[r_f2].insert(r_g2,Rj)
                Seed.append(fgjob0)
        Cmax_m = []
        for i in range(len(Seed)):
            Cmax_m.append(mathCmax(Process, Seed[i], Jobs, Set_t, num_fac))
        Cmax_sorted, indices = sort_and_get_indices(Cmax_m)
        Seed_sorted = sortPop(Seed, indices)
        Pop_fgjob=Seed_sorted[0:50]
        Fjobm=copy.deepcopy(Pop_fgjob[0])
        Fjobm,Cmaxm=Nsinsert_TSIG1(Fjobm,num_fac,Set_t,Jobs,Process)
        end_time=time.time()

        #第二阶段
    while end_time-time_start<limit_time:
        Cmax_m = []
        num_seedlist = []
        Seed = []
        Pop_fgjob.append(Fjobm)


        for i in range(len(Pop_fgjob)):
            Cmax_m.append(mathCmax(Process, Pop_fgjob[i], Jobs, Set_t, num_fac))
        Cmax_sorted, indices = sort_and_get_indices(Cmax_m)
        Pop_fgjob = sortPop(Pop_fgjob, indices)
        for i in Cmax_sorted:
            if Cmax_sorted[-1] - Cmax_sorted[0]==0:
                num_seed=1
            else:
                num_seed = int(1 + ((Cmax_sorted[-1] - i) / (Cmax_sorted[-1] - Cmax_sorted[0])) * 9)  # 参数可更改
            num_seedlist.append(num_seed)
        for i in range(len(Pop_fgjob)):

            for j in range(num_seedlist[i]):
                fgjob0 = copy.deepcopy(Pop_fgjob[i])
                r1 = random.randint(0, len(fgjob0) - 1)
                rr1 = random.randint(0, len(fgjob0[r1]) - 1)
                while len(fgjob0[r1][rr1]) <= 1:
                    rr1 = random.randint(0, len(fgjob0[r1]) - 1)
                rrr_se = random.sample(range(len(fgjob0[r1][rr1])), 2)
                fgjob0[r1][rr1][rrr_se[0]], fgjob0[r1][rr1][rrr_se[1]] = fgjob0[r1][rr1][rrr_se[1]], fgjob0[r1][rr1][
                    rrr_se[0]]
                Seed.append(fgjob0)
        Cmax_m = []
        for i in range(len(Seed)):
            Cmax_m.append(mathCmax(Process, Seed[i], Jobs, Set_t, num_fac))
        Cmax_sorted, indices = sort_and_get_indices(Cmax_m)
        Seed_sorted = sortPop(Seed, indices)
        Pop_fgjob=Seed_sorted[0:50]
        Fjobm = copy.deepcopy(Pop_fgjob[0])
        Fjobm,Cmaxm=Lsinsert_TSIG1(Fjobm,num_fac,Set_t,Jobs,Process)
        end_time=time.time()
    return Fjobm,Cmaxm
def DIWO2(Process,Jobs,Set_t,num_fac,Fjobm,limit_time):
    Pop_gjob = []
    for _ in range(9):
        Jobs1 = copy.deepcopy(Jobs)
        shuffle_nested_lists(Jobs1)
        Pop_gjob.append(Jobs1)
    Pop_fgjob = initialization0(Pop_gjob, Process, Jobs, Set_t, num_fac)

    time_start = time.time()
    end_time = time.time()
    #limit_time=len(Process[0])*len(Jobs)*5
    while end_time - time_start < limit_time:
        Cmax_m = []
        num_seedlist = []
        Seed = []
        Pop_fgjob.append(Fjobm)
        for i in range(len(Pop_fgjob)):
            Cmax_m.append(mathCmax(Process, Pop_fgjob[i], Jobs, Set_t, num_fac))
        Cmax_sorted, indices = sort_and_get_indices(Cmax_m)
        Pop_fgjob = sortPop(Pop_fgjob, indices)
        for i in Cmax_sorted:
            if Cmax_sorted[-1] - Cmax_sorted[0]==0:
                num_seed=1
            else:
                num_seed = int(1 + ((Cmax_sorted[-1] - i) / (Cmax_sorted[-1] - Cmax_sorted[0])) * 9)  # 参数可更改
            num_seedlist.append(num_seed)
        for i in range(len(Pop_fgjob)):
            fgjob = copy.deepcopy(Pop_fgjob[i])
            Cmax_f = []
            for j0 in range(num_fac):
                Cmax0 = sum(mathallD(Process, fgjob[j0], Jobs, Set_t))
                Cmax_f.append(Cmax0)
            for j in range(num_seedlist[i]):
                fgjob = copy.deepcopy(Pop_fgjob[i])
                r_f1 = random.randint(0, num_fac - 1)
                r_g1 = random.randint(0, len(fgjob[r_f1]) - 1)
                Rj = fgjob[r_f1].pop(r_g1)
                r_f2 = random.randint(0, num_fac - 1)
                r_g2 = random.randint(0, len(fgjob[r_f2]) - 1)
                fgjob[r_f2].insert(r_g2, Rj)#对组操作完毕
                fgjob0 = copy.deepcopy(fgjob)
                r1 = random.randint(0, len(fgjob0) - 1)
                rr1 = random.randint(0, len(fgjob0[r1]) - 1)
                while len(fgjob0[r1][rr1]) <= 1:
                    rr1 = random.randint(0, len(fgjob0[r1]) - 1)
                rrr_se = random.sample(range(len(fgjob0[r1][rr1])), 2)
                fgjob0[r1][rr1][rrr_se[0]], fgjob0[r1][rr1][rrr_se[1]] = fgjob0[r1][rr1][rrr_se[1]], fgjob0[r1][rr1][
                    rrr_se[0]]
                Seed.append(fgjob0)

        Cmax_m = []
        for i in range(len(Seed)):
            Cmax_m.append(mathCmax(Process, Seed[i], Jobs, Set_t, num_fac))
        Cmax_sorted, indices = sort_and_get_indices(Cmax_m)
        Seed_sorted = sortPop(Seed, indices)
        Pop_fgjob = Seed_sorted[0:50]
        Fjobm = copy.deepcopy(Pop_fgjob[0])
        Fjobm, Cmaxm = Nsinsert_TSIG1(Fjobm, num_fac, Set_t, Jobs, Process)
        Fjobm, Cmaxm = Lsinsert_TSIG1(Fjobm, num_fac, Set_t, Jobs, Process)
        end_time = time.time()
    return Fjobm, Cmaxm
def DIWO3(Process,Jobs,Set_t,num_fac,Fjobm,limit_time):
    Pop_gjob = []
    for _ in range(9):
        Jobs1 = copy.deepcopy(Jobs)
        shuffle_nested_lists(Jobs1)
        Pop_gjob.append(Jobs1)
    Pop_fgjob = initialization0(Pop_gjob, Process, Jobs, Set_t, num_fac)

    time_start = time.time()
    end_time = time.time()
    #limit_time=len(Process[0])*len(Jobs)*5
    a=0.9
    while end_time-time_start<limit_time:
        ap = random.uniform(0, 1)
        if ap<a:
            Cmax_m = []
            num_seedlist = []
            Seed = []
            Pop_fgjob.append(Fjobm)
            for i in range(len(Pop_fgjob)):
                Cmax_m.append(mathCmax(Process, Pop_fgjob[i], Jobs, Set_t, num_fac))
            Cmax_sorted, indices = sort_and_get_indices(Cmax_m)
            Pop_fgjob = sortPop(Pop_fgjob, indices)
            for i in Cmax_sorted:
                if Cmax_sorted[-1] - Cmax_sorted[0] == 0:
                    num_seed = 1
                else:
                    num_seed = int(1 + ((Cmax_sorted[-1] - i) / (Cmax_sorted[-1] - Cmax_sorted[0])) * 9)  # 参数可更改
                num_seedlist.append(num_seed)
            for i in range(len(Pop_fgjob)):
                for j in range(num_seedlist[i]):
                    fgjob = copy.deepcopy(Pop_fgjob[i])
                    r_f1 = random.randint(0, num_fac - 1)
                    r_g1 = random.randint(0, len(fgjob[r_f1]) - 1)
                    Rj = fgjob[r_f1].pop(r_g1)
                    r_f2 = random.randint(0, num_fac - 1)
                    r_g2 = random.randint(0, len(fgjob[r_f2]) - 1)
                    fgjob[r_f2].insert(r_g2, Rj)
                    Seed.append(fgjob)
            Cmax_m = []
            for i in range(len(Seed)):
                Cmax_m.append(mathCmax(Process, Seed[i], Jobs, Set_t, num_fac))
            Cmax_sorted, indices = sort_and_get_indices(Cmax_m)
            Seed_sorted = sortPop(Seed, indices)
            Pop_fgjob = Seed_sorted[0:50]
            Fjobm = copy.deepcopy(Pop_fgjob[0])
            Fjobm, Cmaxm = Nsinsert_TSIG1(Fjobm, num_fac, Set_t, Jobs, Process)

        #第二阶段
        else:
            Cmax_m = []
            num_seedlist = []
            Seed = []
            Pop_fgjob.append(Fjobm)
            for i in range(len(Pop_fgjob)):
                Cmax_m.append(mathCmax(Process, Pop_fgjob[i], Jobs, Set_t, num_fac))
            Cmax_sorted, indices = sort_and_get_indices(Cmax_m)
            Pop_fgjob = sortPop(Pop_fgjob, indices)
            for i in Cmax_sorted:
                if Cmax_sorted[-1] - Cmax_sorted[0] == 0:
                    num_seed = 1
                else:
                    num_seed = int(1 + ((Cmax_sorted[-1] - i) / (Cmax_sorted[-1] - Cmax_sorted[0])) * 9)  # 参数可更改
                num_seedlist.append(num_seed)
            for i in range(len(Pop_fgjob)):
                for j in range(num_seedlist[i]):
                    fgjob0 = copy.deepcopy(Pop_fgjob[i])
                    r1 = random.randint(0, len(fgjob0) - 1)
                    rr1 = random.randint(0, len(fgjob0[r1]) - 1)
                    while len(fgjob0[r1][rr1]) <= 1:
                        rr1 = random.randint(0, len(fgjob0[r1]) - 1)
                    rrr_se = random.sample(range(len(fgjob0[r1][rr1])), 2)
                    fgjob0[r1][rr1][rrr_se[0]], fgjob0[r1][rr1][rrr_se[1]] = fgjob0[r1][rr1][rrr_se[1]], \
                                                                             fgjob0[r1][rr1][
                                                                                 rrr_se[0]]
                    Seed.append(fgjob0)
            Cmax_m = []
            for i in range(len(Seed)):
                Cmax_m.append(mathCmax(Process, Seed[i], Jobs, Set_t, num_fac))
            Cmax_sorted, indices = sort_and_get_indices(Cmax_m)
            Seed_sorted = sortPop(Seed, indices)
            Pop_fgjob = Seed_sorted[0:50]
            Fjobm=copy.deepcopy(Pop_fgjob[0])
            Fjobm, Cmaxm = Lsinsert_TSIG1(Fjobm, num_fac, Set_t, Jobs, Process)
        end_time = time.time()

    return Fjobm,Cmaxm
def employed_bee(fgjob,Process,Jobs,Set_t,num_fac,Pop_new):
    for _ in range(5):
        zsbs = 0
        while zsbs == 0:
            r1 = random.randint(0, num_fac - 1)
            fgjob0 = copy.deepcopy(fgjob)
            Cnow = sum(mathallD(Process, fgjob0[r1], Jobs, Set_t))
            rr1 = random.randint(0, len(fgjob0[r1]) - 1)
            rr2 = random.randint(0, len(fgjob0[r1]) - 1)
            if rr1 != rr2:
                fgjob0[r1][rr1], fgjob0[r1][rr2] = fgjob0[r1][rr2], fgjob0[r1][rr1]
            Cnew = sum(mathallD(Process, fgjob0[r1], Jobs, Set_t))
            if Cnew >= Cnow:
                zsbs = 1
                Pop_new.append(fgjob0)
        zsbs = 0

        while zsbs == 0:  # 同工厂组插入
            r1 = random.randint(0, num_fac - 1)
            fgjob0 = copy.deepcopy(fgjob)
            Cnow =sum(mathallD(Process, fgjob0[r1], Jobs, Set_t))
            rr1 = random.randint(0, len(fgjob0[r1]) - 1)
            rj1 = fgjob0[r1].pop(rr1)
            rr2 = random.randint(0, len(fgjob0[r1]) - 1)
            fgjob0[r1].insert(rr2, rj1)
            Cnew = sum(mathallD(Process, fgjob0[r1], Jobs, Set_t))

            if Cnew >= Cnow:
                zsbs = 1
                Pop_new.append(fgjob0)
        zsbs = 0

        while zsbs == 0:  # 异工厂组交换
            r_se = random.sample(range(num_fac), 2)
            fgjob0 = copy.deepcopy(fgjob)
            Cnow0 = sum(mathallD(Process, fgjob0[r_se[0]], Jobs, Set_t))
            Cnow1 = sum(mathallD(Process, fgjob0[r_se[1]], Jobs, Set_t))
            Cnow = max(Cnow0, Cnow1)
            rr1 = random.randint(0, len(fgjob0[r_se[0]]) - 1)
            rr2 = random.randint(0, len(fgjob0[r_se[1]]) - 1)
            fgjob0[r_se[0]][rr1], fgjob0[r_se[1]][rr2] = fgjob0[r_se[1]][rr2], fgjob0[r_se[0]][rr1]
            Cnew0 = sum(mathallD(Process, fgjob0[r_se[0]], Jobs, Set_t))
            Cnew1 = sum(mathallD(Process, fgjob0[r_se[1]], Jobs, Set_t))
            Cnew = max(Cnew0, Cnew1)

            if Cnew >= Cnow:
                zsbs = 1
                Pop_new.append(fgjob0)

        zsbs = 0
        while zsbs == 0:  # 异工厂组插入
            r_se = random.sample(range(num_fac), 2)
            while len(fgjob[r_se[0]]) <= 1 and len(fgjob[r_se[1]]) <= 1:
                r_se = random.sample(range(num_fac), 2)
            fgjob0 = copy.deepcopy(fgjob)
            Cnow0 = sum(mathallD(Process, fgjob0[r_se[0]], Jobs, Set_t))
            Cnow1 = sum(mathallD(Process, fgjob0[r_se[1]], Jobs, Set_t))
            Cnow = max(Cnow0, Cnow1)
            rr1 = random.randint(0, len(fgjob0[r_se[0]]) - 1)
            rr2 = random.randint(0, len(fgjob0[r_se[1]]) - 1)
            fgjob0[r_se[1]].insert(rr2, fgjob0[r_se[0]].pop(rr1))
            Cnew0 = sum(mathallD(Process, fgjob0[r_se[0]], Jobs, Set_t))
            Cnew1 = sum(mathallD(Process, fgjob0[r_se[1]], Jobs, Set_t))
            Cnew = max(Cnew0, Cnew1)

            if Cnew >= Cnow:
                zsbs = 1
                Pop_new.append(fgjob0)

        zsbs = 0

        while zsbs == 0:  # 组内工件交换
            r1 = random.randint(0, num_fac - 1)
            fgjob0 = copy.deepcopy(fgjob)
            Cnow = sum(mathallD(Process, fgjob0[r1], Jobs, Set_t))
            rr1 = random.randint(0, len(fgjob0[r1]) - 1)
            while len(fgjob0[r1][rr1]) <= 1:
                rr1 = random.randint(0, len(fgjob0[r1]) - 1)
            rrr_se = random.sample(range(len(fgjob0[r1][rr1])), 2)
            fgjob0[r1][rr1][rrr_se[0]], fgjob0[r1][rr1][rrr_se[1]] = fgjob0[r1][rr1][rrr_se[1]], fgjob0[r1][rr1][
                rrr_se[0]]
            Cnew = sum(mathallD(Process, fgjob0[r1], Jobs, Set_t))
            if Cnew >= Cnow:
                zsbs = 1
                Pop_new.append(fgjob0)

        zsbs = 0

        while zsbs == 0:  # 组内工件插入
            r1 = random.randint(0, num_fac - 1)
            fgjob0 = copy.deepcopy(fgjob)
            Cnow = sum(mathallD(Process, fgjob0[r1], Jobs, Set_t))
            rr1 = random.randint(0, len(fgjob0[r1]) - 1)
            while len(fgjob0[r1][rr1]) <= 1:
                rr1 = random.randint(0, len(fgjob0[r1]) - 1)
            rrr_1 = random.randint(0, len(fgjob0[r1][rr1]) - 1)
            rj1 = fgjob0[r1][rr1].pop(rrr_1)
            rrr_2 = random.randint(0, len(fgjob0[r1][rr1]) - 1)
            fgjob0[r1][rr1].insert(rrr_2, rj1)
            Cnew = sum(mathallD(Process, fgjob0[r1], Jobs, Set_t))
            if Cnew >= Cnow:
                zsbs = 1
                Pop_new.append(fgjob0)

    return Pop_new
def onlooker_bee(Pop_fgjob,Process,Jobs,Set_t,num_fac):
    Pop_new=[]
    Cmax_m = []
    for _ in range(20):
        r_lists=random.sample(Pop_fgjob,4)
        Cmaxm=float('inf')
        for i in r_lists:
            Cmax=mathCmax(Process, i,Jobs,Set_t,num_fac)
            if Cmax<Cmaxm:
                Cmaxm=Cmax
                fgjobm=i
        Pop_new=employed_bee(fgjobm,Process,Jobs,Set_t,num_fac,Pop_new)
    merged_lists = []

    # 遍历lists1和lists2中的子列表
    for sublist in Pop_new + Pop_fgjob:
        # 如果子列表不在结果列表中，就将其添加到结果列表中
        if sublist not in merged_lists:
            merged_lists.append(sublist)
    Pop_fgjob=merged_lists
    for i in range(len(Pop_fgjob)):
        Cmax_m.append(mathCmax(Process, Pop_fgjob[i], Jobs, Set_t, num_fac))
    Cmax_sorted,indices=sort_and_get_indices(Cmax_m)
    Pop_fgjob=sortPop(Pop_fgjob, indices)
    Fgjob=copy.deepcopy(Pop_fgjob)
    Pop_nfj=Fgjob[:30]
    return Pop_nfj
def scout_bee(Pop_fgjob,Process,Jobs,Set_t,num_fac):
    fjobm=copy.deepcopy(Pop_fgjob[0])
    zsbs=0
    while zsbs==0:
        Cmax0=mathCmax(Process,fjobm,Jobs,Set_t,num_fac)
        fjobm,Cmaxm = Nsinsert_TSIG1(fjobm, num_fac, Set_t, Jobs, Process)
        if Cmaxm>=Cmax0:
            zsbs=1
    zsbs=0
    while zsbs==0:
        Cmax0=mathCmax(Process,fjobm,Jobs,Set_t,num_fac)
        fjobm,Cmaxm = Nsswap_TSIG2(fjobm, num_fac, Set_t, Jobs, Process)

        if Cmaxm>=Cmax0:
            zsbs = 1
    zsbs = 0
    while zsbs == 0:
        Cmax0 = mathCmax(Process, fjobm, Jobs, Set_t, num_fac)
        fjobm ,Cmaxm= Lsinsert_TSIG1(fjobm, num_fac, Set_t, Jobs, Process)
        if Cmaxm >= Cmax0:
            zsbs = 1
    Cmaxm=mathCmax(Process,fjobm,Jobs,Set_t,num_fac)
    return fjobm,Cmaxm
def DABC(Process,Jobs,Set_t,num_fac,Fjobm,limit_time):
    Pop_gjob=[]
    for _ in range(50):
        Jobs1 = copy.deepcopy(Jobs)
        shuffle_nested_lists(Jobs1)
        Pop_gjob.append(Jobs1)
    time_start=time.time()
    end_time=time.time()
    Pop_fgjob = initialization0(Pop_gjob, Process, Jobs, Set_t, num_fac)
    Pop_fgjob.append(Fjobm)
    while end_time-time_start<limit_time:
        Pop_new=[]

        for i in Pop_fgjob:
            Pop_new=employed_bee(i,Process,Jobs,Set_t,num_fac,Pop_new)
        merged_lists = []
        for sublist in Pop_new + Pop_fgjob:
            # 如果子列表不在结果列表中，就将其添加到结果列表中
            if sublist not in merged_lists:
                merged_lists.append(sublist)
        Pop_fgjob = merged_lists
        Pop_fgjob = onlooker_bee(Pop_fgjob, Process, Jobs, Set_t, num_fac)
        Fjobm, Cmaxm = scout_bee(Pop_fgjob, Process, Jobs, Set_t, num_fac)
        end_time=time.time()

    return Fjobm, Cmaxm



for dio in range(100,150):#len(Txt_nac)
    Fbest = []
    Cbest = []
    C0 = []
    C1 = []
    C2 = []
    C3 = []
    C4 = []
    C5 = []
    C6 = []
    C7 = []
    C8 = []
    CALL = []
    Csum = []
    Maxc=float('inf')
    Jobs = copy.deepcopy(Txt_Jobs[dio])
    Process = copy.deepcopy(Txt_t[dio])
    Set_t = copy.deepcopy(Txt_Set[dio])
    num_fac = copy.deepcopy(Txt_fac[dio])
    time_limit=100
    for _ in range(5):
        QM = [[0 for _ in range(len(Process))] for _ in range(len(Process) + 1)]
        RM = [[0 for _ in range(len(Process))] for _ in range(len(Process) + 1)]
        QM = newset(QM, Jobs)
        Cmaxm = float('inf')
        QMc = copy.deepcopy(QM)
        Fjobm = initialization_Gjob(Process, Jobs, Set_t, num_fac)
        Fjobm, Cmaxm= QMFG(QM, Jobs, Process, Set_t, num_fac, Fjobm, Cmaxm,time_limit)

        C0.append(Cmaxm)
        if Cmaxm<Maxc:
            Maxc=Cmaxm
            FM=Fjobm

    for _ in range(5):

        QM = [[0 for _ in range(len(Process))] for _ in range(len(Process) + 1)]
        QM = newset(QM, Jobs)
        Fjobm = initialization_Gjob0(Process, Jobs, Set_t, num_fac)
        Cmaxm = float('inf')
        Fjobm, Cmaxm = iterated(Jobs, Process, Set_t, num_fac, Fjobm, Cmaxm,time_limit)
        C1.append(Cmaxm)
        if Cmaxm<Maxc:
            Maxc=Cmaxm
            FM = Fjobm


    for _ in range(5):
        start_time = time.time()
        fgjob_l, Cmax_l = alg_mCCEA(Jobs, Process, Set_t, num_fac,time_limit)
        end_time = time.time()
        run_time = end_time - start_time
        C2.append(min(Cmax_l))
        if min(Cmax_l) < Maxc:
            Maxc = min(Cmax_l)
            FM = Fjobm


    for _ in range(5):

        fgjob = initialization_Gjob0(Process, Jobs, Set_t, num_fac)
        Fjobm, Cmaxm = CoIG(Jobs, Process, Set_t, num_fac, fgjob,time_limit)
        C3.append(Cmaxm)
        if Cmaxm < Maxc:
            Maxc = Cmaxm
            FM = Fjobm



    for _ in range(5):

        fgjob = initialization_Gjob0(Process, Jobs, Set_t, num_fac)
        Fjobm, Cmaxm = DIWO1(Process,Jobs,Set_t,num_fac,fgjob,time_limit)
        C5.append(Cmaxm)
        if Cmaxm < Maxc:
            Maxc = Cmaxm
            FM = Fjobm

    for _ in range(5):

        fgjob = initialization_Gjob(Process, Jobs, Set_t, num_fac)
        Fjobm, Cmaxm = DIWO2(Process,Jobs,Set_t,num_fac,fgjob,time_limit)

        C6.append(Cmaxm)
        if Cmaxm < Maxc:
            Maxc = Cmaxm
            FM = Fjobm

    for _ in range(5):

        fgjob = initialization_Gjob(Process, Jobs, Set_t, num_fac)
        Fjobm, Cmaxm = DIWO3(Process,Jobs,Set_t,num_fac,fgjob,time_limit)
        C7.append(Cmaxm)
        if Cmaxm < Maxc:
            Maxc = Cmaxm
            FM = Fjobm
    Cbest.append(Maxc)
    Fbest.append(FM)
    CALL=[C0,C1,C2,C3,C5,C6,C7]
    for z in range(len(CALL)):
        Csum.append(((sum(CALL[z])/5-Maxc)/Maxc)*100)
    data={'Alg': ['QM', 'TIG', 'CCEA','CIG','DIWO1','DIWO2','DIWO3'],
    dio: Csum}
    print(data)

    with open('Fbest.txt', 'a') as file:
        line = ', '.join(map(str, Fbest))
        file.write( str(dio)+', '+line + ', ' + str(Maxc) + '\n')
    existing_data = pd.read_excel('output_0.xlsx')

    combined_data = existing_data.merge(pd.DataFrame(data), on='Alg', how='left')
    combined_data.to_excel('output_0.xlsx', index=False)
    '''existing_data = pd.read_excel('output_0.xlsx')
    # 将 'Alg' 列用作索引以匹配现有的数据
    

    df.set_index('Alg', inplace=True)
    combined_data.set_index('Alg', inplace=True)
    combined_data = pd.concat([combined_data, df], axis=0, sort=False)

    combined_data.to_excel('output_1.xlsx', index=False)'''


    #with pd.ExcelWriter(f'output_{dio}.xlsx', engine='openpyxl') as writer:
        #sheet_name = 'Sheet1'
        #df.to_excel(writer, sheet_name=sheet_name, index=False)
print(Csum)






