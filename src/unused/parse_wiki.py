import time

def Time(s):
    print("At %d s, " % (time.time() - start) + s)

fpath = "wikipedia/enwiki-20181020-pages-articles-multistream.xml"

chunk_size = 1000000000
global start
start = time.time()
with open(fpath,'r') as f:
    nb_chunk = 0
    while (1):
        nb_chunk += 1
        foutpath = "cleaned_wiki/cleaned-wiki-chunk_%d.txt"%(nb_chunk)
        with open(foutpath, "w") as fout:
            chunk = f.readlines(chunk_size)
            print(len(chunk))
            if not chunk:
                print("hhh")
                break
            print("chunk No (%d,%d)"%(nb_chunk, len(chunk))
            # print chunk
            # chunkstr = ''.join([s.strip() for s in chunk])
            chunkstr = ''.join(chunk)
            Time("a chunk is joined into a long html")

            # method 1 with html2text
            text_maker = html2text.HTML2Text()
            # text_maker.ignore_links = True
            text = text_maker.handle(chunkstr.decode('utf8'))
            # text = html2text.html2text(chunkstr.decode('utf8'))

            # print text
            Time("long text extracted from long html")

            # method 2
            # text = remove_tags(chunkstr)
            # print type(text)
            cleaned_text = parseXmlStopStemRem1by1(text,[])


            # build_graph_with(cleaned_text)

            # print cleaned_text
            # only for test purpose

            # fout.write(text.encode("utf8"))
            # fout.write("="*80)
            fout.write(cleaned_text)
            # break

