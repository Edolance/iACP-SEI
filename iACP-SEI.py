import torch
import esm
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
import time
import pandas as pd
from rich.progress import track
import warnings
import joblib
from sklearn.model_selection import StratifiedKFold
from mlxtend.plotting import plot_decision_regions
from mlxtend.classifier import StackingClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
 
warnings.filterwarnings('ignore')
import preprocessing.fasta as fasta

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

def ESM2_t33_Embed(fastaFile,outFile):
    T0=time.time()
    model, alphabet = pretrained.load_model_and_alphabet("./esm_model/esm2_t33_650M_UR50D.pt")
    model.eval()
    
    if torch.cuda.is_available() :
        model = model.cuda()
        print("Transferred model to GPU")
        
    inData=fasta.fasta2csv(fastaFile)
    SEQ_=inData["Seq"]
    PID_=inData["PID"]
    CLASS_=inData["Class"]

    feats=[]
    feats_labels=[]
    
    dataset = FastaBatchedDataset.from_file(fastaFile)
    batches = dataset.get_batch_indices(1, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
    )
    
    print(f"Read fasta with {len(dataset)} sequences")


    with torch.no_grad():  
        for batch_idx, (labels, strs, toks) in track(enumerate(data_loader),"computing...."):
            #print(
             #   f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            #)
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=[33], return_contacts=0)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            
            for i, label in enumerate(labels):
                #print(labels,i,label)
                 
                result = {"label": label}
                truncate_len = min(1022, len(strs[i]))
                
                if "mean" in ['mean']:
                    result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
            feats.append(result["mean_representations"][33].numpy())
            feats_labels.append(strs)

    print(len(feats))
    #print(feats[1])
    
    data=pd.DataFrame(feats)
    
    id_label=pd.DataFrame(feats_labels,columns=["ID_seq"])

    feats_vec_pd=pd.concat([id_label,data],axis=1,ignore_index=True)
    col=["ID_seq"]
    for i in range(1280):
        col.append("ESM2_t33_650M_UR50D_F"+str(i+1))
     
    feats_vec_pd.columns=col
    print(feats_vec_pd.shape)
    feats_vec_pd.to_csv(outFile+".vec.csv",index=False)
    return feats_vec_pd

def ESM2_t36_Embed(fastaFile,outFile):
    T0=time.time()
    model, alphabet = pretrained.load_model_and_alphabet("./esm_model/esm2_t36_3B_UR50D.pt")
    model.eval()
    
    if torch.cuda.is_available() :
        model = model.cuda()
        print("Transferred model to GPU")
        
    inData=fasta.fasta2csv(fastaFile)
    SEQ_=inData["Seq"]
    PID_=inData["PID"]
    CLASS_=inData["Class"]

    feats=[]
    feats_labels=[]
    
    dataset = FastaBatchedDataset.from_file(fastaFile)
    batches = dataset.get_batch_indices(1, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(1022), batch_sampler=batches
    )
    
    print(f"Read fasta with {len(dataset)} sequences")


    with torch.no_grad():  
        for batch_idx, (labels, strs, toks) in track(enumerate(data_loader),"computing...."):
            #print(
             #   f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            #)
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=[36], return_contacts=0)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            
            for i, label in enumerate(labels):
                #print(labels,i,label)
                 
                result = {"label": label}
                truncate_len = min(1022, len(strs[i]))
                
                if "mean" in ['mean']:
                    result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
            feats.append(result["mean_representations"][36].numpy())
            feats_labels.append(strs)

    print(len(feats))
    #print(feats[1])
    
    data=pd.DataFrame(feats)
    
    id_label=pd.DataFrame(feats_labels,columns=["ID_seq"])

    feats_vec_pd=pd.concat([id_label,data],axis=1,ignore_index=True)
    col=["ID_seq"]
    for i in range(1280):
        col.append("ESM2_t33_650M_UR50D_F"+str(i+1))
     
    feats_vec_pd.columns=col
    print(feats_vec_pd.shape)
    feats_vec_pd.to_csv(outFile+".vec.csv",index=False)
    return feats_vec_pd

def getTrainedModel(m_type):
    path_="./TrainedModel/ACP."+m_type

    ml_model=joblib.load(path_+".ML.Model.joblib")
    data_STD=joblib.load(path_+".data.STD.joblib")
    featureName=pd.read_csv(path_+".FeatureName.csv",header=0)
    feat_col=featureName.columns

    return ml_model,data_STD,feat_col

 
    


if  __name__ == "__main__":

    import argparse
    p = argparse.ArgumentParser(description='Run Machine Learning Classifiers for Anticaner Peptides identification Using ESM embbeding Features.')  
    p.add_argument('-inFasta', type=str, help="Input peptides sequence in FASTA format" ,  default=None)
    p.add_argument('-m', type=str, help="Alt, Main, Merged for model trained based on Alt / Main / Merged Dataset, respectively" , default=None)
    p.add_argument('-out',  type=str,  help="Output the predicted results", default="Results")
    args = p.parse_args()

    inFasta=args.inFasta
    out=args.out
    
    if args.m=="Alt":
        ml_model, data_STD,featureName=getTrainedModel("Alt")
        featvec=ESM2_t33_Embed(inFasta,args.out)
        X_test=featvec[featureName]
        X_test=data_STD.transform(X_test)
        y_pred=ml_model.predict(X_test[:,:70])
        y_pred_proba=ml_model.predict_proba(X_test[:,:70])
        outpd=pd.DataFrame(y_pred,columns=["pred_label"])
        outpd["Positive_Probability(/%)"]=y_pred_proba[:,1]*100
        outpd=pd.concat([featvec.iloc[:,0],outpd],axis=1,ignore_index=True)
        outpd.columns=["seq","Pred_label(1 for Pos,0 for Neg)","Positive_Probability(/%)"]
        outpd.to_csv(args.out+".pred_results.csv")
        
    elif args.m=="Main":
        ml_model, data_STD,featureName=getTrainedModel("Main")
        featvec=ESM2_t36_Embed(inFasta,args.out)
        X_test=featvec[featureName]
        X_test=data_STD.transform(X_test)
        y_pred=ml_model.predict(X_test[:,:124])
        y_pred_proba=ml_model.predict_proba(X_test[:,:124])
        outpd=pd.DataFrame(y_pred,columns=["pred_label"])
        outpd["Positive_Probability(/%)"]=y_pred_proba[:,1]*100
        outpd=pd.concat([featvec.iloc[:,0],outpd],axis=1,ignore_index=True)
        outpd.columns=["seq","Pred_label(1 for Pos,0 for Neg)","Positive_Probability(/%)"]
        outpd.to_csv(args.out+".pred_results.csv")
    elif args.m=="Merged":
        ml_model, data_STD,featureName=getTrainedModel("Merged")
        featvec=ESM2_t33_Embed(inFasta,args.out)
        X_test=featvec[featureName]
        X_test=data_STD.transform(X_test)
        y_pred=ml_model.predict(X_test[:,:1280])
        y_pred_proba=ml_model.predict_proba(X_test[:,:1280])
        outpd=pd.DataFrame(y_pred,columns=["pred_label"])
        outpd["Positive_Probability(/%)"]=y_pred_proba[:,1]*100
        outpd=pd.concat([featvec.iloc[:,0],outpd],axis=1,ignore_index=True)
        outpd.columns=["seq","Pred_label(1 for Pos,0 for Neg)","Positive_Probability(/%)"]
        outpd.to_csv(args.out+".pred_results.csv")
    else:
        print("Trainded model de not exist!")
       
    