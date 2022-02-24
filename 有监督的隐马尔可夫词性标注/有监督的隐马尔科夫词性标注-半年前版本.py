import numpy as np
import copy

DATA = 'data/train.conll'
OUT = 'data/myresult.txt'
#-------------练习-------------#
def search(o,V):
    for i in range(len(V)):
        if o==V[i]:
            return i
    return -1
#前向算法
def for_algo(V,Π,A,B,O):
    for_p_l=list()
    o=O[0]
    u = search(o,V)
    for_p = np.multiply(Π, B).T[u]
    for_p_l.append(for_p)

    for i in O[1:]:
        u=search(i,V)
        for_p = np.multiply(for_p * A, B.T[u])
        for_p_l.append(for_p)
    p = np.sum(for_p,axis=1)
    return for_p_l,p
#后向算法
def bac_algo(V,Π,A,B,O):
    bac_p_l=list()
    bac_p = np.mat([1]*len(A))
    for i in range(len(O)-1,0,-1):
        o=O[i]
        u = search(o, V)
        bac_p = np.multiply(A * B, bac_p.T).T[u]
        bac_p_l.insert(0,bac_p)

    o=O[0]
    u = search(o, V)
    bac_p = np.multiply(np.multiply(Π,B).T[u],bac_p)
    p=np.sum(bac_p,axis=1)
    bac_p_l.insert(0,bac_p)
    return bac_p_l,p
#t时刻状态为i的概率
def qi_p_t(t,qi,for_p_l,bac_p_l,Q):
    p = np.multiply(for_p_l[t],bac_p_l[t])
    for i in range(len(Q)):
        if Q[i] == qi:
            p = p[0][i]/np.sum(p,axis=1)[0][0]
    return p
#t时刻状态为i，t+1到j的概率
def qij_p_t(t,qi,qj,o,for_p_l,bac_p_l,Q,A,V):
    i=0
    j=0
    m=0
    for k in range(len(Q)):
        if Q[k] == qi:
            i=k
        if Q[k]==qj:
            j=k
    for k in range(len(V)):
        if V[k]==o:
            m=k
    p1 = for_p_l[t][i]*A[i][j]*B[j][m]*bac_p_l[t][j]
    p2 = np.sum(np.multiply(np.multiply(np.multiply(for_p_l[t].T,A),B.T[m]),bac_p_l[t].T))
    p=p1/p2
    return p
#O观测序列下状态为qi出现的期望
def p_qi(qi,for_p_l,bac_p_l,Q):
    p = 0
    for i in range(len(O)):
        p+=qi_p_t(i,qi,for_p_l,bac_p_l,Q)
    return p
#O观测序列下由状态i转移的期望值
def p_qi_(qi,for_p_l,bac_p_l,Q):
    p=0
    for i in range(len(O)-1):
        p+=qi_p_t(i,qi,for_p_l,bac_p_l,Q)
    return p
#O观测序列下由状态i转移到状态j的期望值
def p_qi_j(qi,qj,o,for_p_l,bac_p_l,Q,A,V,O):
    p=0
    for i in range(len(O)-1):
        p+=qij_p_t(i,qi,qj,o,for_p_l,bac_p_l,Q,A,V)
    return p
def kexi_ti(t,i,for_p_l,bac_p_l,A,B,V,result,O,Π):
    if result[t][i]!=0:
        return result[t][i][0],result[t][i][1],result[t][i][2]
    my_i=0
    kexi=0
    my_o = 0  # 求o下标
    for oi in range(len(V)):
        if V[oi] == O[t]:
            my_o = oi
    if t==0:
        kexi=Π[i]*B[i].T[my_o]
        result[t][i]=[-1,i,kexi]
        return -1,i,kexi
    else:
        temp = 0
        for j in range(len(A)):
            my_i,my_j,q=kexi_ti(t-1,j,for_p_l,bac_p_l,A,B,V,result,O,Π)

            q=q*A[j].T[i]*B[i].T[my_o]
            if q>kexi:
                kexi = q
                temp=j
        result[t][i]=[temp,i,kexi]
        return my_i,i,kexi
def Viterbi(V,Π,A,B,O):
    for_p_l,for_p = for_algo(V,Π,A,B,O)
    bac_p_l,bac_p = bac_algo(V,Π,A,B,O)
    max = 0
    k=0
    result = list([0] * len(A) for h in range(len(O)))
    for i in range(len(A)):
        kexi_ti(len(O)-1, i,for_p_l, bac_p_l, A,B, V, result,O,Π)
        q=result[-1][i][-1]
        if q>max:
            max=q
            k=i
    I=list()
    I.insert(0,result[-1][k][1])
    i=result[-1][k][0]
    for h in range(len(result)-2,-1,-1):
        I.insert(0, result[h][i][1])
        i=result[h][i][0]
    return I
def mysearch(a,mylist):
    for i in range(len(mylist)):
        if mylist[i]==a:
            return i
    return -1
def myread_cut(path):
    f = open(path, 'r',encoding='utf-8')
    text = f.readlines()
    result = list()
    temp = list()
    for i in range(len(text)):
        if len(text[i])==1:
            result.append(copy.deepcopy(temp))
            temp.clear()
            continue
        temp.append(text[i].split('\t')[1])
    return result
def myread_answer(path):
    f = open(path, 'r',encoding='utf-8')
    text = f.readlines()
    result = list()
    temp = list()
    for i in range(len(text)):
        if len(text[i])==1:
            result.append(copy.deepcopy(temp))
            temp.clear()
            continue
        temp.append(text[i].split('\t')[3])
    return result
def myread_conll(path):
    f = open(path,'r',encoding = 'utf-8')
    text = f.readlines()

    for i in range(len(text)-1,-1,-1):
        if len(text[i])==1:
            text.pop(i)

    length = len(text)

    phrase_character_list = [0]*length
    phrase_list = [0]*length
    character_list = [0]*length
    #将词组和词性放入p_c_l,将词组和词性分别放入p_l,c_l
    for line in range(length):
        text[line] = text[line].split('\t')
        phrase = text[line][1]
        character = text[line][3]
        phrase_character_list[line]=[phrase,character]
        phrase_list[line] = phrase
        character_list[line] = character
    #去重
    phrase_list = list(set(phrase_list))
    character_list = list(set(character_list))

    f.close()

    return phrase_character_list,phrase_list,character_list
def myparameter(path):
    phrase_character_list, phrase_list, character_list=myread_conll(path)
    N=len(character_list)
    M=len(phrase_list)

    A = [[0]*N for i in range(N)]
    B = [[0]*M for i in range(N)]
    i1 = mysearch(phrase_character_list[0][1],character_list)
    for k in range(len(phrase_character_list)-1):
        #获得构建隐藏状态转移矩阵数据
        j1 = mysearch(phrase_character_list[k+1][1],character_list)
        A[i1][j1]+=1
        i1=j1
        #获得构建可观测状态发射矩阵数据
        i2 = mysearch(phrase_character_list[k+1][0],phrase_list)
        B[j1][i2]+=1


    #构建可观测状态发射矩阵，隐藏状态转移矩阵，先验概率Π
    my_A = np.mat(A)
    my_B = np.mat(B)
    Π = [0] * N
    A_count = np.sum(my_A,axis = 1)
    B_count = np.sum(my_B,axis = 1)
    A_count_sum = np.sum(A_count)
    for i in range(len(Π)):
        Π[i] = float(A_count[i]/A_count_sum)
    a=1
    for i in range(N):
        for j in range(N):
            A[i][j] = float((A[i][j]+a)/(A_count[i]+a*N))
        for j in range(M):
            B[i][j] = float((B[i][j]+a)/(B_count[i]+a*N))
    my_A = np.mat(A)
    my_B = np.mat(B)
    Π=np.mat(Π).T
    return phrase_list,character_list,Π,my_A,my_B
def mywrite(path,result):
    f=open(path,'w')
    temp = ""
    for i in result:
        for j in i:
            temp+=j+" "
        temp+="\n"
    f.write(temp)
    f.close()
def myread_result(path):
    f = open(path, 'r')
    text = f.readlines()
    result=list()
    for i in text:
        result.append(i.split())
    return result
def evaluate(cut,answer):
    right_count=0
    count=0
    for i in range(len(cut)):
        for j in range(len(cut[i])):
            if cut[i][j]==answer[i][j]:
                right_count+=1
            count+=1
    TA = right_count/count
    return TA


def test():
    V, Q, Π, A, B = myparameter(DATA)
    cut = myread_cut(DATA)
    answer = myread_answer(DATA)
    result_list=[0]*len(cut)
    for i in range(len(cut)):
        I = Viterbi(V,Π,A,B,cut[i])
        result = copy.deepcopy(I)
        for j in range(len(I)):
            result[j]=Q[I[j]]
        result_list[i]=copy.deepcopy(result)
    mywrite(OUT,result_list)

def myevaluate():
    answer = myread_answer(DATA)
    result_list = myread_result(OUT)
    TA = evaluate(result_list,answer)
    print(TA)

test()
myevaluate()