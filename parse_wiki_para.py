import html2text
from data_cleaner import *
from scrapy.utils.markup import remove_tags
import pprint
import time
from multiprocessing import cpu_count,Pool
from functools import partial


def Time(s):
    print "At %d s, " % (time.time() - start) + s

def jump_n_chunk(n,chunk_size):
    for i in range(n):
        if i % 10 == 0:
            print i
        _ = f.readlines(chunk_size)


fpath = "wikipedia/enwiki-20181020-pages-articles-multistream.xml"

global start
start = time.time()

chunk_size = 10000000
processes = cpu_count()
processes = 2
print "processes number = %d" % processes
pool = Pool(processes=processes)

with open(fpath, 'r') as f:
    nb_chunk = 0
    jump_n = 2424
    jump_n_chunk(jump_n,chunk_size)
    nb_chunk = jump_n
    while (1):
        nb_chunk += 1
        foutpath = "cleaned_wiki/cleaned-wiki-chunk_%d.txt" % nb_chunk
        with open(foutpath, "w") as fout:
            chunk = f.readlines(chunk_size)
            print len(chunk)
            if not chunk:
                print "hhh"
                break
            print "\nchunk No (%d,%d)" % (nb_chunk, len(chunk))
            # print chunk
            # chunkstr = ''.join(chunk)
            Time("a chunk is stored into a list")

            # method 1 with html2text
            # text_maker = html2text.HTML2Text()
            # text = text_maker.handle(chunkstr.decode('utf8'))
            chunk = [t.decode('utf8') for t in chunk]
            text_by_lines = pool.map(html2text.html2text,chunk)
            # print(text_by_lines)

            # text = html2text.html2text(chunkstr.decode('utf8'))

            # print text
            Time("tags removed from each line of HTML")

            # method 2
            # text = remove_tags(chunkstr)
            # print type(text)
            # cleaned_text = parseXmlStopStemRem1by1(text, [])
            partial_work = partial(parseXmlStopStemRem1by1, unique_words=[])
            cleaned_text_by_lines = pool.map(partial_work, text_by_lines)
            Time("text cleaned line by line")

            cleaned_text_by_lines = [t for t in cleaned_text_by_lines if t != '']
            cleaned_text = ' '.join(cleaned_text_by_lines)
            Time("clean text joined together")

            # build_graph_with(cleaned_text)

            # print cleaned_text
            # only for test purpose

            # _ = [fout.write(t.encode("utf8")) for t in text_by_lines]
            # fout.write("="*80+"\n")
            fout.write(cleaned_text)
            if nb_chunk == 2426:
                break

