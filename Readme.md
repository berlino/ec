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
##### LARC Dataset 
The program synthesis experiments use **LARC (Language-annotated Abstraction and Reasoning Corpus)** (Sec. 3 of the paper), a dataset containing human descriptions of the ARC[https://github.com/fchollet/ARC] (Chollet 2019)  inductive reasoning tasks gathered via a 2-participant communication game. The full language dataset collection procedure is described in the main paper and released separately. This repository also contains the following pre-processed versions of this dataset, which can be used directly to reproduce the results of the reported experiments:

1. ARC tasks. Our experiments use the training dataset (n=400 tasks) from the original ARC repository. JSON files containing the input/output examples for each task and a task_id are in `arc_data/data/training/<task_id>.json`.
2. Human natural language task annotations. Sentence-parsed annotations for each task from the human experiment are in `data/arc/language/sentences/language.json`, in a dict mapping from task_id to an array of strings for each task.
3. Natural language DSL function annotations. Additional expert-written annotations for each _function_ in the DSL, used in the _pseudo-annotation_ training procedure (Sec. 5.1), are available in `data/arc/primitiveNamesToDescriptions.json`. This maps each function in the DSL to both a human readable name and a human readable short gloss (stored in a `[name, gloss]` tuple); we use the gloss for our experiments.

