## Results in my environment

See [youtube](https://youtu.be/dO3ZcdEOb5E) and [arxiv](https://arxiv.org/abs/2101.07625)

## Important dependencies

- https://github.com/stack-of-tasks/pinocchio
- https://github.com/ikalevatykh/panda3d_viewer
- https://github.com/Gepetto/example-robot-data

## Installation

If you use anaconda, just run
1. `conda env create -f pino.yaml`
1. `conda activate pino`

## Usage

To run the test program, `python test.py`
with three arguments: `--task --name --n_resume`

To evaluate the results obtained by the test program, `python eval.py`

Or just run `bash test.bash`

## Citation

```latex
@misc{kobayashi2021mirrordescent,
    title={Mirror-Descent Inverse Kinematics for Box-constrained Joint Space},
    author={Taisuke Kobayashi},
    year={2021},
    eprint={2101.07625},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
    url={https://arxiv.org/abs/2101.07625},
}
```
