# DNA Classification


Requirements
* Docker


Run baseline single hidden layer experiments
* With GPU `make run-net-baseline GPU=true CUDA_VISIBLE_DEVICES=0`
* Without GPU `make run-net-baseline GPU=false`

Run cnn+blstm experiments
* With GPU `make run-train GPU=true CUDA_VISIBLE_DEVICES=0`
* Without GPU `make run-train GPU=false`


Run cnn+blstm cross dataset experiments
* With GPU `make run-train-cross GPU=true CUDA_VISIBLE_DEVICES=0`
* Without GPU `make run-train-cross GPU=false`


Obs.:
In case of memory problems with experiments, run on GPU mode.
