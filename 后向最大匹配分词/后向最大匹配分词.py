import copy
#-------------------------数据处理--------------------------#
DATA = 'data/train.conll'
DICT = 'data/word.dict'
TEXT = 'data/data.txt'
OUTPUT = 'data/data.out'


def write_dict(data_path, dict_path):
    max_num = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        words = {line.split()[1] for line in f if len(line) > 1}
    with open(dict_path, 'w', encoding='utf-8') as f:
        for word in words:
            f.write(word+"\n")
            if max_num < len(word):
                max_num = len(word)
    return max_num


def write_data(data_path, text_path):
    with open(data_path, 'r', encoding='utf-8') as data:
        with open(text_path, 'w', encoding='utf-8') as f:
            for line in data:
                if len(line) > 1:#判空
                    word = line.split()[1]
                    f.write(word)
                else:
                    f.write("\n")


def mytran(text_path, dict_path, out_path, max_len):
    with open(dict_path, 'r', encoding='utf-8') as f:
        words = {line.strip() for line in f}
    with open(text_path, 'r', encoding='utf-8') as text:
        with open(out_path, 'w', encoding='utf-8') as out:
            for line in text:
                i = 0
                while i < len(line.strip()):
                    j = i + max_len
                    while j > i + 1 and line[i:j] not in words:
                        j -= 1
                    out.write(line[i:j] + "\n")
                    i = j


#--------------------------评价程序--------------------------------#
def evaluate(data_path, out_path):
    with open(out_path, 'r', encoding='utf-8') as out:
        with open(data_path, 'r', encoding='utf-8') as data:
            x = [line.split()[1] for line in data if len(line) > 1]
            y = [line.strip() for line in out]
    result_count = len(x)
    answer_count = len(y)

    cret_count = 0
    i=0
    j=0
    while i < result_count and j < answer_count:
        if x[i] == y[j]:
            cret_count += 1
        else:
            x_temp, y_temp = x[i], y[j]
            while x_temp != y_temp:
                if len(x_temp) > len(y_temp):
                    j += 1
                    y_temp += y[j]
                elif len(x_temp) < len(y_temp):
                    i += 1
                    x_temp += x[i]
        i += 1
        j += 1

    P=cret_count/result_count
    R=cret_count/answer_count
    F=P*R*2/(P+R)
    print("正确率为：",P)
    print("召回率为：", R)
    print("F:", F)
    return F


max_num = write_dict(DATA, DICT)
write_data(DATA, TEXT)
mytran(TEXT,DICT,OUTPUT,max_num)
evaluate(DATA, OUTPUT)
