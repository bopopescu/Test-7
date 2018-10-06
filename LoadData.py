import re
import os
from bs4 import BeautifulSoup
from urllib2 import urlopen
import codecs
from glob import glob

def _getRawDataFromText():
   # currentDir=os.getcwd()
   # os.chdir('/Users/shibshankarroy/PycharmProjects/Test/Data')
   # print currentDir
   # os.chdir('/Users/shibshankarroy/PycharmProjects/Test')
   # print currentDir
    book_filenames = glob(os.path.join("Data","got*.txt"))
    corpus_raw = u""
    for book_filename in book_filenames:
        with codecs.open(book_filename, "r", "utf-8") as book_file:
            corpus_raw += book_file.read()
    #cleanData = re.sub("[^a-zA-Z0-9,.!?]", " ", corpus_raw)
    return corpus_raw
    #got1File = open(os.path.join("Data","got1.txt"),"r")
    #data1 = got1File.read()
    #got2File = open(os.path.join("Data", "got2.txt"), "r")
    #data2 = got2File.read()
    #got3File = open(os.path.join("Data", "got3.txt"), "r")
    #data3 = got3File.read()
    #got4File = open(os.path.join("Data","got4.txt"),"r")
    #data4 = got4File.read()
    #got5File = open(os.path.join("Data", "got5.txt"), "r")
    #data5 = got1File.read()
    #data = data1.__add__(data2.__add__(data3.__add__(data4.__add__(data5))))
    #cleanData=re.sub("[^a-zA-Z0-9,.!?]"," ",data)
    #return cleanData




def _getRawDataFromWeb():
    url ="https://en.wikipedia.org/wiki/The_Indian_Express"
    html=urlopen(url)
    bs =BeautifulSoup(html,"lxml")
    paras = bs.find_all("p")
    data=""
    for para in paras:
       data = data+ para.text
    cleanData = re.sub("[^a-zA-Z0-9,.!?]", " ", data)
    return cleanData

_getRawDataFromText()
#  "\"\n",
       # "#for each book, read it, open it un utf 8 format, \n",
        #"#add it to the raw corpus\n",
        #"for book_filename in book_filenames:\n",
        #"    print(\"Reading '{0}'...\".format(book_filename))\n",
      #  "    with codecs.open(book_filename, \"r\", \"utf-8\") as book_file:\n",
       # "        corpus_raw += book_file.read()\n",
        #"    print(\"Corpus is now {0} characters long\".format(len(corpus_raw)))\n",
        #"    print()"