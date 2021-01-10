---
layout: post # 使用的布局（不需要改）
title: biopython # 标题
subtitle: python中的生物信息学分析 #副标题
date: 2021-01-06 # 时间
author: CHY # 作者
header-img: img/wallhaven-6oqy97.jpg #这篇文章标题背景图片
catalog: true # 是否归档
tags: #标签
  - 生物信息
---

本节主要记录biopython库的学习，学习资源来自于生信修炼手册公众号，仅做个人学习使用。

#### biopython简介
biopython可用于常规的生信分析处理：
1. 对常用的文件格式，比如fasta, blast等，进行读写
2. 对blast, clustalw等常用软件的集成
3. 对NCBI, SwissPort, PDB等常用生物信息学数据库的检索和解析
4. 进化树的构建
5. 基因组数据的可视化

其中存在多个针对不同任务开发的子模块
1. Bio.Seq, 提供了Seq类，即生物学序列对象，最常见的就是碱基或者核酸序列，比如fasta文件中保存的序列
2. Bio.SeqRecord, 提供了SeqRecord类，包含了序列的注释信息，比如fasta文件中的序列标识符
3. Bio.SeqIO, 提供了parse方法，来读取不同格式的序列文件，比如fasta/genebank等格式
4. Bio.Align, 提供了MultipleSeqAlignment对象，以及读取多序列比输出结果文件的方法
5. Bio.Blast, 提供了运行blast比对软件的方法，以及解析blast输出结果的方法
6. Bio.Entrez, 提供了NCBI Entrez 系统的接口，可以查询，检索，下载， 解析数据库中的内容
7. Bio.SwissPort, 提供了Swiss-prot数据库的接口，可以查询，检索，下载， 解析数据库中的内容
8. Bio.PDB, 提供了PDB数据库的接口，可以查询，检索，下载， 解析数据库中的内容
9. Bio.Phylo, 提供了查看系统发育树和可视化的各种方法
10. Bio.Graphics, 提供了基因组数据的可视化功能

#### 使用biopython处理序列数据
biopython中存在三个子模块可用于处理序列数据
1. Bio.Seq 表示最原始的序列对象，提供了序列的格式化，反向互补，碱基计数等基本功能
2. Bio.SeqRecore 表示序列记录，在序列对象的基础上，进一步添加了序列的id, 名称，属性等各种注释信息
3. Bio.SeqIO 用于读取特定的文件格式，返回 SeqRecord对象

```
from Bio.Seq import Seq
my_seq = Seq('ATCGTACGATCT')
my_seq

# 切片
my_seq[1]
my_seq[1:3]
my_seq[::-1]
# 小写转换
my_seq.lower()
# 大写转换
my_seq.upper()
# split, 序列分隔
my_seq.split('A')
# join, 序列连接
my_seq2 = Seq('ACGACTGACTAGCT')
Seq('NNN').join([my_seq, my_seq2])
# 格式化
'id:1,seq:{}'.format(my_seq)
# 互补
my_seq.complement()
# 反向互补
my_seq.reverse_complement()
# 转录
my_seq.transcribe()
# 翻译
my_seq.translate()
```
```
# Bio.SeqRecord在序列的基础上，进一步存储了相关的注释信息
from Bio.SeqRecord import SeqRecord
my_seq = Seq('AGCTACGT')
my_seqrecord = SeqRecord(my_seq)
my_seqrecord
# 多种信息查看
my_seqrecord.seq
my_seqrecord.id
my_seqrecord.name
my_seqrecord.description
```
```
# Bio.SeqIO用于文件的读写
from Bio import SeqIO
for seq in SeqIO.parse('input.fasta', 'fasta'):
    print(seq.id, seq.seq)
for seq in SeqIO.parse('input.gb', 'genebank'):
    print(seq.id, seq.seq)

# 读入为list
records = list(SeqIO.parse('input.fasta', 'fasta'))
records[0]

# 格式转换
records = SeqIO.parse("input.gb", "genbank")
SeqIO.write(records, "out.fasta", "fasta")
count = SeqIO.convert("input.gb", "genbank", "out.fasta", "fasta")
```

#### 序列比对在biopython中的处理
局部比对最经典的代表是blast, 全局比对则用于多序列比对。
```
# 读取多序列比对结果 Bio.AlignIO
from Bio import AlignIO
alignment = AlignIO.parse('clustal.out', 'clustal')
print(alignment)
for i in alignment:
     print(i.id)

# 输出多序列比对结果
alignments = AlignIO.parse("aln.fasta", "fasta")
AlignIO.write(alignments, "aln.clustal", "clustal")
# 格式可转换
count = AlignIO.convert("aln.fasta", "fasta", "align.clustal", "clustal")

# 运行多序列比对程序
from Bio.Align.Applications import ClustalwCommandline
cline = ClustalwCommandline("clustalw2", infile="input.fasta")

# blast运行
# 联网状态下，调用NCBI网站的blast程序
# 传统的文件读取, 适合fasta格式
from Bio.Blast import NCBIWWW
fasta_string = open("input.fasta").read()
result_handle = NCBIWWW.qblast("blastn", "nt", fasta_string)
# Bio.SeqIO读取，适合fasta,genebank等格式
record = SeqIO.read("input.fasta", format="fasta")
result_handle = NCBIWWW.qblast("blastn", "nt", record.format('fasta'))

# 本地运行需要构建对应数据库
from Bio.Blast.Applications import NcbiblastxCommandline
blastx_cline = NcbiblastxCommandline(query="query.fasta", db="nr", evalue=0.001, outfmt=5, out="output.xml")
stdout, stderr = blastx_cline()

# 解析blast的输出
# biopython中blast默认的输出格式为xml
from Bio.Blast import NCBIXML
blast_records = NCBIXML.parse(result_handle)
E_VALUE_THRESH = 0.001
for blast_record in blast_records:
     for alignment in blast_record.alignments:
        for hsp in alignment.hsps:
             if hsp.expect < E_VALUE_THRESH:
                 print '****Alignment****'
                 print 'sequence:', alignment.title
                 print 'length:', alignment.length
                 print 'e value:', hsp.expect
                 print hsp.query[0:75] + '...'
                 print hsp.match[0:75] + '...'
                 print hsp.sbjct[0:75] + '...'
```

#### 使用biopython查询NCBI数据库
biopython将NCBI的API接口Eutils工具进行了封装，通过Bio.Entrez子模块，可以在python环境中与NCBI进行交互。
```
# EInfo查看数据库的基本信息
Entrez.email = "hongyuchen@zju.edu.cn"
handle = Entrez.einfo()
record = Entrez.read(handle)
record
record.keys()
record['DbList']

# EInfo也查询某个特定的数据库的信息
handle = Entrez.einfo(db='pubmed')
record = Entrez.read(handle)
record.keys()
record['DbInfo'].keys()
record['DbInfo']['DbName']
record['DbInfo']['MenuName']
record['DbInfo']['Description']

# ESearch检索数据库
handle = Entrez.esearch(db="pubmed", term="cnv-seq")
record = Entrez.read(handle)
record.keys()
record["IdList"]

# EPost上传待查询的ID到NCBI服务器
id_list = ["19304878", "18606172", "16403221", "16377612", "14871861", "14630660"]
search_results = Entrez.read(Entrez.epost("pubmed", id=",".join(id_list)))
webenv = search_results["WebEnv"]
query_key = search_results["QueryKey"]
webenv
query_key

# ESummary下载对应的摘要信息
handle = Entrez.esummary(db="pubmed", id="33255631")
record = Entrez.read(handle)
record
record[0].keys()
record[0]['Item']
record[0]['Title']

# EFetch下载数据库中的内容
handle = Entrez.efetch(db="nucleotide", id="186972394", rettype="gb", retmode="text")
context = handle.read()
with open('out.gb', 'w') as fp:
    fp.write(context)

# ELink数据库之间的交叉查询
record = Entrez.read(Entrez.elink(dbfrom="gene", db="clinvar", id="7157"))
record[0]['LinkSetDb'][0]['Link'][0]

# EGQuery统计检索项在各个数据库中检索到的条目
handle = Entrez.egquery(term="biopython")
record = Entrez.read(handle)
for row in record["eGQueryResult"]:
     print(row["DbName"], row["Count"])

# ESpell自动校正拼写错误
handle = Entrez.espell(term="biopythooon")
record = Entrez.read(handle)
record.keys()
record['Query']
record['CorrectedQuery']
```

#### 进化树在biopython中的可视化
通过Bio.Phylo子模块，可以方便的访问和展示树状结构中的信息。
```
# 读取对应文件
from Bio import Phylo
tree = Phylo.read('tree.newick', 'newick')
tree

# 查看树状结构
print(tree)

# 可视化
tree.rooted=True
Phylo.draw(tree)

# 修改分支颜色
tree = tree.as_phyloxml()
tree.root.color = "gray"
mcra = tree.common_ancestor({"name":"E"}, {"name":"F"})
mcra.color = "salmon"
tree.clade[0, 1].color = "blue"
Phylo.draw(tree)
```

#### 使用biopython可视化染色体和基因元件
通过BiolGraphics子模块可以对基因组结构进行可视化，支持线性和圈图两种可视化方式。其中，基因组结构信息存储在genebank格式的文件中，首先通过Bio.SeqIO读取结构信息，然后通过Bio.Graphics模块进行可视化。
```
# 示例数据：https://www.ncbi.nlm.nih.gov/nuccore/NC_005816
from reportlab.lib import colors
from reportlab.lib.units import cm
from Bio.Graphics import GenomeDiagram
from Bio import SeqIO
record = SeqIO.read("sequence.gb", "genbank")

# 提取gb文件中的feature信息，构建用于绘图的数据结构
gd_diagram = GenomeDiagram.Diagram("Yersinia pestis biovar Microtus plasmid pPCP1")
gd_track_for_features = gd_diagram.new_track(1, name="Annotated Features")
gd_feature_set = gd_track_for_features.new_set()
for feature in record.features:
     if feature.type != "gene":
         continue
     if len(gd_feature_set) % 2 == 0:
         color = colors.blue
     else:
         color = colors.lightblue
     gd_feature_set.add_feature(feature, color=color, label=True)

# 线性图
gd_diagram.draw(format="linear", orientation="landscape", pagesize='A4', fragments=4, start=0, end=len(record))
gd_diagram.write("plasmid_linear.pdf", "PDF")

# 圈图
gd_diagram.draw(format="linear", orientation="landscape", pagesize='A4', fragments=4, start=0, end=len(record), circle_core=0.7)
gd_diagram.write("plasmid_linear.pdf", "PDF")
```
```
# 染色体图
entries = [("Chr I", 30432563),
            ("Chr II", 19705359),
            ("Chr III", 23470805),
           ("Chr IV", 18585042),
            ("Chr V", 26992728)]
max_len = 30432563
telomere_length = 1000000
chr_diagram = BasicChromosome.Organism()
chr_diagram.page_size = (29.7*cm, 21*cm) #A4 landscape
for name, length in entries:
     cur_chromosome = BasicChromosome.Chromosome(name)
     cur_chromosome.scale_num = max_len + 2 * telomere_length
     start = BasicChromosome.TelomereSegment()
     start.scale = telomere_length
     cur_chromosome.add(start)
     body = BasicChromosome.ChromosomeSegment()
     body.scale = length
     cur_chromosome.add(body)
     end = BasicChromosome.TelomereSegment(inverted=True)
     end.scale = telomere_length
     cur_chromosome.add(end)
     chr_diagram.add(cur_chromosome)
chr_diagram.draw("simple_chrom.pdf", "Arabidopsis thaliana")

# 在染色体上添加注释，标记基因组结构元件在染色体上的分布
chr_diagram = BasicChromosome.Organism()
chr_diagram.page_size = (29.7 * cm, 21 * cm) # A4 landscape
entries = [
     ("Chr I", "NC_003070.gbk"),
     ("Chr II", "NC_003071.gbk"),
     ("Chr III", "NC_003074.gbk"),
     ("Chr IV", "NC_003075.gbk"),
     ("Chr V", "NC_003076.gbk"),
]
max_len = 30432563
telomere_length = 1000000
chr_diagram = BasicChromosome.Organism()
chr_diagram.page_size = (29.7 * cm, 21 * cm) 
for index, (name, filename) in enumerate(entries):
     record = SeqIO.read(filename, "genbank")
     length = len(record)
     features = [f for f in record.features if f.type == "tRNA"]
     for f in features:
         f.qualifiers["color"] = [index + 2]
     cur_chromosome = BasicChromosome.Chromosome(name)
     cur_chromosome.scale_num = max_len + 2 * telomere_length
     start = BasicChromosome.TelomereSegment()
     start.scale = telomere_length
     cur_chromosome.add(start)
     body = BasicChromosome.AnnotatedChromosomeSegment(length, features)
     body.scale = length
     cur_chromosome.add(body)
     end = BasicChromosome.TelomereSegment(inverted=True)
     end.scale = telomere_length
     cur_chromosome.add(end)
     chr_diagram.add(cur_chromosome)
chr_diagram.draw("tRNA_chrom.pdf", "Arabidopsis thaliana")
```

#### 使用biopython解析kegg数据库
原理在于利用KEGG数据库提供的API接口。在biopython中，通过Bio.KEGG模块，对kegg官方的API进行了封装，允许在python环境中使用kegg API。
```
# 下载数据
from Bio.KEGG import REST
pathway = REST.kegg_get('hsa00010')

# 查询内容转换为纯文本
pathway = REST.kegg_get('hsa00010')
res = pathway.read().split("\n")
res[0]
res[1]

# 结果解析
from Bio.KEGG import REST
request = REST.kegg_get("ec:5.4.2.2")
open("ec_5.4.2.2.txt", "w").write(request.read())
records = Enzyme.parse(open("ec_5.4.2.2.txt"))
record = list(records)[0]
record
record.classname
record.entry

# KEGG数据筛选案例
from Bio.KEGG import REST
human_pathways = REST.kegg_list("pathway", "hsa").read()
repair_pathways = []
for line in human_pathways.rstrip().split("\n"):
     entry, description = line.split("\t")
     if "repair" in description:
         repair_pathways.append(entry)

repair_pathways
repair_genes = []
for pathway in repair_pathways:
     pathway_file = REST.kegg_get(pathway).read()
     current_section = None
     for line in pathway_file.rstrip().split("\n"):
         section = line[:12].strip()
         if not section == "":
             current_section = section
         if current_section == "GENE":
             gene_identifiers, gene_description = line[12:].split("; ")
             gene_id, gene_symbol = gene_identifiers.split()
             if not gene_symbol in repair_genes:
                 repair_genes.append(gene_symbol)

repair_genes
```