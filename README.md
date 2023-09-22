# NLP_Practice
Here is BIT NLP process course resource. 

I do my own implementation of the model part from sratch which in ./model/pfn1.py. I successfully reproduced [A Partition Filter Network for Joint Entity and Relation Extraction](https://arxiv.org/abs/2108.1220) model, PFN.

REFERENCE : https://github.com/Coopercoppers/PFN#Data-Acquisition-and-Preprocessing.


# DATASET

First of all, you should first make dir named "data" and go to [Here](https://github.com/Coopercoppers/PFN#Data-Acquisition-and-Preprocessing) to get dataset.

We choose dataset "NYT" for example as follows.

# Train
Here give u a example about the 
```
python .\main.py --data NYT --do_train --do_eval --embed_mode bert_cased --batch_size 2 --lr 0.00002 
--eval_metric micro --output_file ace_test1
```

# Evaluate

If you want to evalute the model. 

```
python eval.py \
--data ${NYT/WEBNLG/ADE/ACE2005/ACE2004/SCIERC} \
--eval_metric ${micro/macro} \
--model_file ${the path of saved model you want to evaluate. e.g. save/ace_test.pt} \
--embed_mode ${bert_cased/albert/scibert}
```

