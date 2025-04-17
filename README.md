## Requirements
config	0.5.1
networkx	3.1		
nilearn	0.10.4	
numexpr	2.8.6	
numpy	1.24.3
pandas	2.0.3
torch	2.2.2+cu118	
torch-cluster	1.6.3+pt22cu118	
torch-geometric	2.5.3	
torch-scatter	2.1.2+pt22cu118	
torch-sparse	0.6.18+pt22cu118	
torch-spline-conv	1.2.2+pt22cu118	
torchaudio	2.2.2+cu118	
torchvision	0.17.2+cu118	
tqdm	4.66.5
## 1.Download the dataset: select one of the following options to run

```
python download_ABIDE.py --derivatives rois_ho
python download_ABIDE.py --derivatives rois_aal
python download_ABIDE.py --derivatives rois_cc200
python download_ABIDE.py --derivatives rois_tt
python download_ABIDE.py --derivatives rois_dosenbach160
```

## 2.Run the node_connect
Run the framework: different datasets correspond to different instructions:
The following instructions correspond to rois_ho,rois_aal,rois_cc200,rois_tt,rois_dosenbach160
```
python node_connect.py --num_samples 871 --num_nodes_per_sample 111
python node_connect.py --num_samples 871 --num_nodes_per_sample 116
python node_connect.py --num_samples 871 --num_nodes_per_sample 200
python node_connect.py --num_samples 871 --num_nodes_per_sample 97
python node_connect.py --num_samples 871 --num_nodes_per_sample 160
```
## 3.Generate functional connection matrix
```
python connect.py
```
## 4.Generate a population graph
```
python gen_population_graph.py
```
## 5.Extracting features of the brain using SAGNet
```
python SAGNetpool.py
```
## 6.Training or evaluation
```
python main.py
```

## 7.Visualization of results
```
python visualize.py --roc True --embedding True --group=gender/age/sites
```

