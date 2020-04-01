#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
import requests

# 把明星的名字存放在这个文件中
f = open('star_name.txt', 'w', encoding='utf-8')


# 获取明星的名字并保存到文件中
def get_page(pages, pagesize, star_name):
    params = []
    # 设置访问的请求头，包括分页数和明星所在的地区
    for i in range(0, pagesize * pages, pagesize):
        params.append({
            'resource_id': 28266,
            'from_mid': 1,
            'format': 'json',
            'ie': 'utf-8',
            'oe': 'utf-8',
            'query': '明星',
            'sort_key': '',
            'sort_type': 1,
            'stat0': '',
            'stat1': star_name,
            'stat2': '',
            'stat3': '',
            'pn': i,
            'rn': pagesize})


    # 请求的百度接口获取明星的名字
    # 可以通过 百度搜索栏搜索明星，然后通过web控制台对明星那一页进行查询，获取到请求的url、参数格式，响应格式，然后抄下来即可
    url = 'https://sp0.baidu.com/8aQDcjqpAAV3otqbppnN2DJv/api.php'

    x = 0
    # 根据请求头下载明星的名字
    for param in params:
        try:
            # 获取请求数据
            res = requests.get(url, params=param, timeout=50)
            # 把网页数据转换成json数据
            js = json.loads(res.text)
            if not js:
                continue
            # 获取json中的明星数据
            results = js.get('data')[0].get('result')
        except AttributeError as e:
            print('【错误】出现错误：%s' % e)
            continue

        # 从数据中提取明星的名字
        for result in results:
            img_name = result['ename']
            f.write(img_name + '\n',)

        if x % 10 == 0:
            print('第%d页......' % x)
        x += 1


# 从百度上获取明星名字
def get_star_name():
    # 获取三个地区的明星
    names = ['内地', '香港', '台湾']
    # 指定每个地区获取的页数
    #sums = [600, 200, 200]
    sums = [1, 1, 1]
    for i in range(len(names)):
        get_page(sums[i], 6,  names[i])

    f.close()


# 删除可能存在错误的名字，但是也存在错误删除问题
def delete_some_name():
    name_set = set()
    with open('star_name.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 5:
                continue
            name_set.add(line)

    print('筛选后的总人数为：%d' % len(name_set))
    with open('star_name.txt', 'w', encoding='utf-8') as f:
        f.writelines(name_set)


if __name__ == '__main__':
    get_star_name()
    delete_some_name()
