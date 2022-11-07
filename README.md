# Assessment of the potential of CLIP in medical imaging context

This repo contains the code for the different approaches to assess and enhance CLIP performance in zero-shot tast under the medical imaging context. The assessment was performed over OpenI dataset collection, primarily on MedPix dataset. Enhancement was performed over two main approaches: fine-tuning and encoders modification.
## Fine tuning

Brief description of fine tuning approach and experimentation

## Encoders modification

Brief description of encoder modification approach and experimentation

## main.py 

We added a main.py file which allow to reproduce the results from the main experiments of our paper and perform classification over an image of your choice. In order to reproduce of our paper run the following command:
```
python main.py --mode test
```
This will reproduce our best performance approach. In order to However, the ```--experiment``` argument will allow you to reproduce other experiments from the paper. Use as input to this argument the notation on the paper