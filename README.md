# deep-nlp

Prerequisites:

1. Python 3.6+
2. Pytorch 0.4+
3. Matplotlib

As of now contains 2 implementations(refer to src/main):

1. Reading comprehension using a pretrained bert model on the SQUAD data-set (Details and running instructions TBA). Needs Squad V1.1 (or 2.0 but unanswered questions are not supported yet) and pytorch-pretrained-bert (https://github.com/huggingface/pytorch-pretrained-BERT). Note that this uses pytorch's checkpointing feature right now (https://pytorch.org/docs/stable/checkpoint.html) to prevent out-of-memory errors on entry-level GPUs
2. Named Entity Recognition on CoNLL 2003 Data. 


Both have been tried on a CPU and Tesla K80/Tesla V100. #1 will take a lot longer than #2 as the BERT transformer is a huge model.

The src/main srcipts will most likely need some changes if you intend to run them locally.

Still a WIP. I'll be adding refinements on a regular basis

Some issues I intend to tackle in the upcoming check-ins: TBA
