---
layout:     post   				    # 使用的布局（不需要改）
title:      Snakemake				# 标题 
subtitle:   生信分析流程搭建 #副标题
date:       2020-06-16				# 时间
author:     CHY					# 作者
header-img: img/wallhaven-2059.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 生信分析
---

#### 安装snakemake
conda install -c bioconda snakemake

#### 简单snakemake脚本举例
```
# 首先创建两个文件夹
echo "Hello Hongyu" > 1.txt
echo "Hi Hongyu" > 2.txt

# 编写snakefile
rule concat:
    input:
        expand("{file}.txt",file=["1","2"])
    output:
        "merged.txt"
    shell:
        "cat {input} > {output}"

# 执行snakefile
snakemake   # 此时表示在snakefile所在路径下，默认寻找snakefile
```

#### 命令及规则
rule all:  # 除了rule all中定义的文件外最后输出结果不会保存任何中间文件。<br>
wildcards:  # 用来获取通配符匹配到的部分<br>
threads:  # 设置运行分配的线程数<br>
message:  # 使用message参数可以指定每运行到一个rule时,在终端中给出提示信息<br>
priority: # 可用来指定程序运行的优先级，默认为0，eg.priority: 20<br>
log:  # 用来指定生成的日志文件<br>
params:  # 指定程序运行的参数<br>
run:  #在run的缩进区域里面可以输入并执行python代码<br>
scripts:  # 用来执行指定脚本<br>
temp:  # 通过temp方法可以在所有rule运行完后删除指定的中间文件<br>
protected:  # 用来指定某些中间文件是需要保留的<br>
ancient:  # 强制使得结果文件一旦生成就不会再次重新生成覆盖，即便输入文件时间戳已经更新<br>
Rule Dependencies:  # 通过快捷方式指定前一个rule的输出文件为此rule的输入文件<br>
report:  # 将结果嵌入到html文件中进行查看<br>

#### snakemake执行参数
--snakefile, -s 指定Snakefile，否则是当前目录下的Snakefile<br>
--dryrun, -n  不真正执行，一般用来查看Snakefile是否有错<br>
--printshellcmds, -p   输出要执行的shell命令<br>
--reason, -r  输出每条rule执行的原因,默认FALSE<br>
--cores, --jobs, -j  指定运行的核数，若不指定，则使用最大的核数<br>
--force, -f 重新运行第一条rule或指定的rule<br>
--forceall, -F 重新运行所有的rule，不管是否已经有输出结果<br>
--forcerun, -R 重新执行Snakefile，当更新了rule时候使用此命令<br>

#一些可视化命令
$ snakemake --dag | dot -Tpdf > dag.pdf
```
#集群投递
snakemake --cluster "qsub -V -cwd -q 节点队列" -j 10
# --cluster /-c CMD: 集群运行指令
# qusb -V -cwd -q， 表示输出当前环境变量(-V),在当前目录下运行(-cwd), 投递到指定的队列(-q), 如果不指定则使用任何可用队列
# --local-cores N: 在每个集群中最多并行N核
# --cluster-config/-u FILE: 集群配置文件
```

#### 参考链接
https://www.jianshu.com/p/14b9eccc0c0e <br>

