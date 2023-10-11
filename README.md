# ESSAformer
The implementation of ESSAformer, a network for Hyperspectral image super-resolution.
## 2023.10.11 
The core code and trained model on Chikuseix4 are uploaded.
## Enviornment
The code is tested under Pytorch 1.12.1, Python 3.8. And it might works on higer Pytorch version.
## Usage 
1, To use the trained model:
 ```
  checkpoint = torch.load(model_name)
  start_epoch = checkpoint["epoch"]
  net.load_state_dict(checkpoint["model"].state_dict())
```
2, Training  
For training, just use any framework with input shape (B,C,H,W). You can use the preprocessing framework of [SSPSR](https://github.com/junjun-jiang/SSPSR) or [MCnet](https://github.com/qianngli/MCNet) for convenience.
## Dataset
1,[Chikusei](https://naotoyokoya.com/Download.html)  
2,[Cave](https://www.cs.columbia.edu/CAVE/databases/multispectral/)  
3,[Pavia centre](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)  
4,[Harvard](https://vision.seas.harvard.edu/hyperspec/index.html)  
5,[NTIRE2022](https://codalab.lisn.upsaclay.fr/competitions/721#participate-get-data)  
6,[NTIRE2020](https://competitions.codalab.org/competitions/22225#participate-get-data)  
7,[ICVL](http://icvl.cs.bgu.ac.il/hyperspectral/)  

## To do 
1,The higher order implementation of ESSA.(We find that for tiny HSI dataset, first order is sufficient. But for larger dataset, maybe higher order ESSA will bring higher accuracy)  
2,Upload trained model of other 6 datasets.
