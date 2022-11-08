## Results in my environment

See [youtube](https://youtu.be/dO3ZcdEOb5E) and [arxiv](https://arxiv.org/abs/2101.07625)

## Important dependencies

- https://github.com/stack-of-tasks/pinocchio
- https://github.com/ikalevatykh/panda3d_viewer
- https://github.com/Gepetto/example-robot-data

## Installation

If you use anaconda, just run
```bash
conda env create -f mdik.yaml
conda activate mdik
```

If you want to use qpsolvers with time limit,
```bash
git clone https://github.com/kbys-t/qpsolvers.git
```

If you want to use qpoases,
```bash
git clone https://github.com/stephane-caron/qpOASES.git
cd qpOASES
make
pip install -e interfaces/python
```
If you get some errors, please check compile flags (especially when you try to install on mac).

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
