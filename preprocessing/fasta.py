from __future__ import print_function, division
import pandas as pd
import os
 

def parse_stream(f, comment=b'#'):
    name = None
    label = None
    sequence = []
    for line in f:
        if line.startswith(comment):
            continue
        line = line.strip()
        if line.startswith(b'>'):
            if name is not None:
                yield name,label, b''.join(sequence)
            name = line[1:]
            label = str(name).split("|")[-1]
            sequence = []
        else:
            sequence.append(line.upper())
    if name is not None:
        yield name,label,b''.join(sequence)

def fasta2csv(inFasta):

    seq={}
    fr=open(inFasta, 'r')
    for line in fr:
        if line.startswith('>'):     
            name=line.replace(" ","|") 
            name=name.replace(",","|")
            seq[name]=''
        else:
            seq[name]+=line.replace('\n', '')
    fr.close()                           
    
    fw=open('out.fasta', 'w')
    for i in seq.keys():
        fw.write(i)
        fw.write(seq[i]+"\n")        
    fw.close()


    FastaRead=pd.read_csv('out.fasta',encoding='utf-8',header=None,sep =",")
    print(FastaRead.shape)
    
    seqNum=int(FastaRead.shape[0]/2)
    csvFile=open("testFasta.csv","w")
    csvFile.write("PID,Class,Seq\n")
    
    
    for i in range(seqNum):
        name=str(FastaRead.iloc[2*i,0]).replace(">","")
        label=name.split("|")[-1]
        sss=str(FastaRead.iloc[2*i+1,0]).upper()
        csvFile.write(name+","+label+","+sss+"\n")
            
         
    csvFile.close()
    TrainSeqLabel=pd.read_csv("testFasta.csv",header=0)
    
    path="testFasta.csv"
    if os.path.exists(path):
     
        os.remove(path)  

    path='out.fasta'
    if os.path.exists(path):
       os.remove(path) 
    
    return TrainSeqLabel
    
    
#TrainSeqLabel=fasta2csv("temp.fasta")
#TrainSeqLabel.to_csv("temp.csv")
    
    





