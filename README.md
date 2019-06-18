# deep-nlp

Prerequisites:

1. Python 3.6+
2. Pytorch 0.4+
3. Matplotlib

As of now contains 3 complete implementations(refer to src/main):

1. squad_reading_comprehension.py: A model I architected which employs a combination of biLSTMs, Bidirectional-Attention and self-Attention for squad 1.1 (https://rajpurkar.github.io/SQuAD-explorer/). It's a WIP, will post updates here as it's completed.
2. squad_reading_comprehension_with_bert.py: Reading comprehension using a pretrained BERT model on a subset of SQUAD  dataset (Basically SQuAD 2.0 with any questions without answer-text removed, details and running instructions TBA). Needs Squad V1.1 (or 2.0 but unanswered questions are not supported yet and will be skipped) and pytorch-pretrained-bert (https://github.com/huggingface/pytorch-pretrained-BERT). Also skips instances where the combined (question + answer) length is more than the maximum token length supported by BERT (512). Note that this uses pytorch's gradient accumulation feature right now to prevent out-of-memory errors on entry-level GPUs. Also has an option to enable checkpointing (https://pytorch.org/docs/stable/checkpoint.html) but I had some trouble with it so have left it disabled by default and have chosen to rely solely on gradient accumulation instead. I was able to get start-index accuracy of 68.5% and end-index accuracy of 71.5% with this implementation after training for 2 epochs with a small batch-size + gradient accumulation. I haven't tried replicating exactly what Google did with their implementation except for using BertAdam with learning-rate warmup and taking cues from the Pytorch port.

I've included a custom SQuAD reader in datasetreaders.py which can be modified to support SQuAD 2.0 as well.
If you intend to run squad_reading_comprehension*.py locally, make sure the training_data_file_path and dev_data_file_path point to the actual SQuAD dataset files obtainable from https://rajpurkar.github.io/SQuAD-explorer/. I chose to not include them in this repo due to their size, so currently, the noted variables point to a sample file which I've been using locally for quickly debugging the script.
That said though, it's generally not a great idea to run this script with the actual dataset and/or a large batch size without access to a GPU. My results were obtained on a Tesla V100 (Floydhub, notebook included in src/main) and the script took close to an hour to run.
**Please take a look at the script(s) before running them to adjust the batch_size and other params etc. as per your choosing.**      

3. ner.py: Named Entity Recognition on CoNLL 2003 Data. Details TBA, uses FastText embeddings right now but one potential thing to try would be to use BERT embeddings instead (possibly while using BERT in no_grad mode)


Both have been tried on a CPU and Tesla K80/Tesla V100. #2 will take a lot longer than #1 and #3 as the BERT transformer is a huge model. All 3 will default to using a cuda gpu if one is available. This can be changed through a single line of code in each src/main script.

The src/main scripts will most likely need some changes if you intend to run them locally.

Still a WIP. Some of the classes included are supporting infrastructure for more complex tasks but are still being developed and aren't used right now. I'll be adding refinements on a regular basis. Comments/suggestions are welcome. Will very much appreciate letting me know if you spot (or think there's a possibility of) a bug. Thanks

**Special thanks to huggingface for porting BERT to Pytorch. Please refer to https://github.com/huggingface/pytorch-pretrained-BERT to access this awesome implementation.**  

