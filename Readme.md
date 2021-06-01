#  Communicating Natural Programs to Humans and Machines : Leveraging Language for Program Synthesis
This repository is the official implementation for the language-guided program synthesis experiments in **Communicating Natural Programs to Humans and Machines** (Section 5, Leveraging Language for Program Synthesis). The paper is currently under review. This repository and branch is a static branch designed to reproduce the results in the paper.

## Getting Started: Dependencies and Requirements
The experiments in the paper were conducted on an academic supercomputing cluster (Ubuntu machines, 24 parallel CPUs per experiment.) The following setup has been tested on both Ubuntu and Mac (OS Catalina). The codebase is implemented in both Python and OCaml.

##### Install Python 3.7.7 and the Python requirements.
1. We test our implementation on 3.7.7. On Linux, you can create a fresh install (including pip) with:
```
sudo apt-get update; sudo apt-get install software-properties-common; 
sudo add-apt-repository ppa:deadsnakes/ppa; sudo apt-get update; 
sudo apt-get install python3.7; sudo apt install python3.7-dev; 
sudo apt install python3-pip; python3.7 -m pip install pip;
pip install --upgrade setuptools;
```
2. Install the requirements.
```
pip install -r neurips_2021_requirements.txt
```
3. Download the T5 pre-trained language model. At an interactive prompt, run:
```
> import transformers; from transformers import T5EncoderModel; T5EncoderModel.from_pretrained('t5-small')
```

##### Build the OCaml binaries.
The repository contains prebuilt OCaml binaries that should run on most Linux-based machines. However, to build the OCaml binaries from scratch, you can run the following from the root of the repo.
1. Install OCaml.
```
sudo apt install ocaml ; 
sudo apt install opam ; 
opam init; opam update; 
opam switch 4.06.1+flambda ;
eval `opam config env`;
```
2. Install the OCaml requirements.
```
opam depext ppx_jane core re2 yojson vg cairo2 camlimages menhir ocaml-protoc zmq; 
opam install ppx_jane core re2 yojson vg cairo2 camlimages menhir ocaml-protoc zmq;
```
3. Run the following from the directory root to build the binaries.
```
make clean; make
```
