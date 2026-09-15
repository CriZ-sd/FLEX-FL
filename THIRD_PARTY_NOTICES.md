# Third-party notices

The root [MIT license](LICENSE) applies to the original FLEX-FL contributions. Portions adapted from third-party sources retain their applicable upstream terms. The project license does not replace those terms or grant rights to external datasets or the associated paper.

## Code included in this repository

### ResNet implementation

- Local file: `netpool.py`, including `BasicBlock`, `ResNet`, `ResNet18`, and `Resnet50`.
- Source credited in the file: [Pytorch-CNN_Resnet18-CIFAR10](https://www.kaggle.com/code/greatcodes/pytorch-cnn-resnet18-cifar10), by Kaggle user `greatcodes` (display name: Nikhil).
- The cited notebook page identifies its license as **Apache License 2.0**.
- Full license: [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).
- These portions have been adapted into the FLEX-FL model module. The local file carries a modification notice and retains the original source link.

Preserve applicable upstream copyright, attribution, and license notices when redistributing these portions. If the particular upstream revision used includes a NOTICE file or additional source notices, retain the applicable notices as required by Apache-2.0. This attribution is not a claim that the notebook author endorses FLEX-FL.

### VGG implementation

- Local file: `netpool.py`, including `VGG`, `make_layers`, the VGG configuration dictionary, and the `vgg*` factory functions.
- Source credited in the file: [chengyangfu/pytorch-vgg-cifar10](https://github.com/chengyangfu/pytorch-vgg-cifar10).
- Copyright (c) 2017 Cheng-Yang Fu.
- License: **MIT**.
- Full upstream notice and license: [licenses/pytorch-vgg-cifar10-MIT.txt](licenses/pytorch-vgg-cifar10-MIT.txt).

The upstream copyright and permission notice must accompany copies or substantial portions of this implementation.

## Research attribution

The `CIFAR10CNN` definition in `netpool.py` references arXiv:2006.10672v2. This is a research attribution, not evidence of a license for separately published source code. Any additional code copied from an external implementation must retain its own applicable notices.

## Separately installed dependencies

The Python source imports the following packages; their distributions are not bundled in this repository. The links below identify upstream license information, rather than relicensing those packages under FLEX-FL's MIT license.

| Dependency | Upstream license information |
| --- | --- |
| PyTorch (`torch`) | [BSD-style license and contributor notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| torchvision | [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) |
| NumPy | [BSD-3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| SciPy | [BSD-3-Clause](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |

If distributing an environment, container, executable, or package that bundles dependencies, include the licenses and notices for the actual versions and bundled components distributed. Individual binary distributions can contain additional third-party components.

## Datasets and publication

CIFAR-10, CIFAR-100, FEMNIST, and the FLEX paper are separate materials. Their inclusion in experiments or citation in the README does not place them under this repository's license. Obtain them from their respective sources and follow the terms applicable to their use and redistribution.

Upstream license information was reviewed on 2026-09-15. The original imported revisions are not recorded in the supplied source snapshot; maintainers should verify their provenance before release.
