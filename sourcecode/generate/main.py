# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
   file name: main.py
   create time: 2017年06月23日 星期五 16时41分54秒
   author: Jipeng Huang
   e-mail: huangjipengnju@gmail.com
   github: https://github.com/hjptriplebee
'''''''''''''''''''''''''''''''''''''''''''''''''''''
from config import *
import data
import dataUtils
import model
import os

def defineArgs():
    """define args"""
    parser = argparse.ArgumentParser(description = "Chinese_poem_generator.")
    parser.add_argument("-m", "--mode", help = "select mode by 'train' or test or head",
                        choices = ["train", "test", "head"], default = "train")
    return parser.parse_args()

def split_poemTxt(filename):
    directory = "dataset/poetryByPoet/"
    if not os.path.exists(directory):
        os.mkdir(directory)
    f_list = os.listdir(directory)
    print(len(f_list))
    if len(f_list) == 0:
        poems = {}
        ignoreStrings = ['_', '《', '[', '(', '{']
        with open(filename, "r") as f:
            for line in f:  # every line is a poem
                title, author, poem = line.strip().split("::")  # get title and poem
                poem = poem.replace(' ', '')
                if len(poem) < 10 or len(poem) > 512:  # filter poem
                    continue
                if any(ext in poem for ext in ignoreStrings):
                    continue
                poem = '[' + poem + ']'
                if author not in poems.keys():
                    poems[author] = list()
                poems[author].append(poem)
                poems[author].append('\n')
        for key, value in poems.items():
            if len(value) < 100:
                continue
            print(len(value))
            tempFile = open(directory + str(key) + '.txt', 'w+', encoding='utf-8')
            tempFile.writelines(value)
            tempFile.close()

    print('finished')

if __name__ == "__main__":
    train_file = "dataset/poetryTang/poetryTang.txt"
    split_poemTxt(train_file)
    print("按照诗人将数据拆分完成！")
    args = defineArgs()
    rootdir  = os.path.dirname(os.path.realpath(__file__)) + "/"
    path = rootdir + "dataset/poetryByPoet"
    f_list = os.listdir(path)
    # print f_list
    for i in f_list:
        fn, ex = os.path.splitext(i)

        # os.path.splitext():分离文件名与扩展名
        if ex != '.txt':
            continue
        if fn.find("完成_") != -1:
            continue

        type = fn
        trainData = dataUtils.POEMS(type)
        MCPangHu = model.MODEL(trainData, type)
        if args.mode == "train":
            MCPangHu.train()
        else:
            if args.mode == "test":
                poems = MCPangHu.test()
            else:
                characters = input("please input chinese character:")
                poems = MCPangHu.testHead(characters)