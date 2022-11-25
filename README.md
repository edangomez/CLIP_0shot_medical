# Assessment of the potential of CLIP in medical imaging context

This repo contains the code for the different approaches to assess and enhance CLIP performance in zero-shot tast under the medical imaging context. The assessment was performed over OpenI dataset collection, primarily on MedPix dataset. Enhancement was performed over two main approaches: fine-tuning and encoders modification.
## Fine tuning

We perform fine-tuning of CLIP using a weighted loss function that leverage the incluence of text and image embbeding in the process of learning visual representations from text. We fine-tune CLIP with various image encoders to assess the incluence and difference between Visual Transformers and ResNet. Furthermore, we perform experimentation to evaluate its zero-shot capabilities over different labels. After fine-tuning, significant increase in performance over medical domain-specific was achieved 

## Replicability

To replicate our experimental framework, use the ```main.py``` file and configurate current directory with the following intructions (all files and folders required can be found at sojedaa/proyecto path on BCV002). In the same directory path of the ```main.py``` file should be:
- **Model** folder which contains the weights of different fine-tuned architecture variations. 
- **MedCLIP** folder which contains necesary files associated to the MedPix dataset to reproduce our experimental framework.
- **CLIP** repo cloned, since our baseline and experimentation requires original CLIP model.

## main.py 

We added a main.py file which allow to reproduce the results from the main experiments of our paper and perform classification over an image of your choice. In order to reproduce of our paper run the following command:
```
python main.py --mode test
```
This will reproduce our best performance approach. In order to However, the ```--experiment``` argument will allow you to reproduce other experiments from the paper. Use as input to this argument the notation on the paper
